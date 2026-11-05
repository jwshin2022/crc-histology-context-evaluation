#!/usr/bin/env python3
"""Bounded CPU development MLP; cooperative pause exits with status 75.

Never infer donor generalization from this engineering/development runner.
See ../RUNNER_CONTRACT.md for schema, state applicability and qualification.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import sys
import time
import uuid

# Must be set before importing numerical libraries. CPU only; no accelerator init.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCHEMA_VERSION = 1
PAUSE_EXIT = 75
SIGNAL_PAUSE = False


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, obj, *, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        raise FileExistsError(path)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        with open(temporary, "x") as f:
            f.write(canonical(obj) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if immutable and path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
        fsync_dir(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def finite_tree(obj, path="state"):
    if isinstance(obj, torch.Tensor):
        if obj.device.type != "cpu":
            raise ValueError(f"Non-CPU tensor at {path}")
        if obj.is_floating_point() and not bool(torch.isfinite(obj).all()):
            raise ValueError(f"Non-finite tensor at {path}")
    elif isinstance(obj, float) and not math.isfinite(obj):
        raise ValueError(f"Non-finite scalar at {path}")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            finite_tree(value, f"{path}.{key}")
    elif isinstance(obj, (tuple, list)):
        for i, value in enumerate(obj):
            finite_tree(value, f"{path}[{i}]")


def rng_state():
    ns = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"algorithm": ns[0], "keys": torch.from_numpy(ns[1].astype(np.int64)),
                  "position": ns[2], "has_gauss": ns[3], "cached_gaussian": ns[4]},
        "torch_cpu": torch.get_rng_state(),
    }


def restore_rng(state):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns["algorithm"], ns["keys"].numpy().astype(np.uint32),
                         ns["position"], ns["has_gauss"], ns["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"])


def load_data(path):
    before = digest(path)
    with np.load(path, allow_pickle=False) as z:
        required = {"X", "Y", "coords", "train_idx", "val_idx", "row_ids", "gene_names", "donors"}
        if not required.issubset(z.files):
            raise ValueError(f"Missing NPZ arrays: {required - set(z.files)}")
        data = {key: z[key].copy() for key in required}
    if before != digest(path):
        raise ValueError("Input changed while reading")
    x, y, coords = data["X"], data["Y"], data["coords"]
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or min(*x.shape, y.shape[1]) < 1:
        raise ValueError("Require nonempty X[n,d], Y[n,g]")
    n = len(x)
    if coords.shape != (n, 2):
        raise ValueError("coords must have shape [n,2]")
    for name in ("X", "Y", "coords"):
        a = data[name]
        if a.dtype.kind != "f" or not np.isfinite(a).all():
            raise ValueError(f"{name} must contain finite floating-point values")
        data[name] = a.astype(np.float32)
        if not np.isfinite(data[name]).all():
            raise ValueError(f"{name} overflows float32")
    for name, length in (("row_ids", n), ("gene_names", y.shape[1])):
        a = data[name]
        if a.shape != (length,) or a.dtype.kind not in "US":
            raise ValueError(f"{name} must be non-object strings of length {length}")
        a = a.astype(str)
        if np.unique(a).size != length or np.any(a == ""):
            raise ValueError(f"{name} must be unique, nonempty strings")
        data[name] = a
    for name in ("train_idx", "val_idx"):
        a = data[name]
        if a.ndim != 1 or not len(a) or a.dtype.kind not in "iu":
            raise ValueError(f"{name} must be a nonempty integer vector")
        if (a < 0).any() or (a >= n).any() or np.unique(a).size != len(a):
            raise ValueError(f"Invalid, duplicate, or out-of-bounds {name}")
        data[name] = a.astype(np.int64)
    roles = np.concatenate([data["train_idx"], data["val_idx"]])
    if len(roles) != n or np.unique(roles).size != n:
        raise ValueError("train_idx and val_idx must be disjoint and cover all rows")
    if data['donors'].shape != (n,) or data['donors'].dtype.kind not in 'US' or np.any(data['donors']==''):
        raise ValueError('donors must be nonempty string identifiers for every row')
    if set(data['donors'][data['train_idx']]) & set(data['donors'][data['val_idx']]):
        raise ValueError('Donors overlap training and validation')
    return data, before


class Head(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(128, output_dim))

    def forward(self, x):
        return self.net(x)


def version_binding():
    lock = Path(__file__).resolve().parents[1] / "requirements.lock.txt"
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": str(np.__version__), "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(),
            "torch_build_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
            "requirements_lock_sha256": digest(lock) if lock.exists() else None,
            "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "device": "cpu", "dtype": "float32", "num_workers": 0,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def checked_load(path, expected_sha):
    if digest(path) != expected_sha:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    finite_tree(state)
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unknown checkpoint schema")
    return state


class Checkpoints:
    def __init__(self, run):
        self.run = run
        self.directory = run / "checkpoints"
        self.directory.mkdir(exist_ok=True)
        self.pointer = run / "CURRENT.json"

    def save(self, state):
        finite_tree(state)
        prior = json.loads(self.pointer.read_text()) if self.pointer.exists() else None
        if prior:
            checked_load(self.run / prior["current"]["path"], prior["current"]["sha256"])
        generation = 1 if prior is None else prior["generation"] + 1
        name = f"g{generation:06d}-u{state['update']:08d}-{state['phase']}.pt"
        path = self.directory / name
        if path.exists():
            raise FileExistsError(f"Immutable checkpoint exists: {path}")
        temporary = self.directory / ("." + name + "." + uuid.uuid4().hex)
        try:
            with open(temporary, "xb") as f:
                torch.save(state, f)
                f.flush()
                os.fsync(f.fileno())
            checksum = digest(temporary)
            verified = checked_load(temporary, checksum)
            if verified["bindings"] != state["bindings"] or verified["update"] != state["update"]:
                raise ValueError("Checkpoint read-back state mismatch")
            os.rename(temporary, path)
            fsync_dir(self.directory)
        finally:
            if temporary.exists():
                temporary.unlink()
        current = {"path": str(path.relative_to(self.run)), "sha256": checksum,
                   "bytes": path.stat().st_size}
        atomic_json(path.with_suffix(".json"), current, immutable=True)
        pointer = {"generation": generation, "current": current,
                   "previous_good": prior["current"] if prior else None}
        atomic_json(self.pointer, pointer)
        # Read-back the published artifact and both pointer/reference files.
        if json.loads(self.pointer.read_text()) != pointer:
            raise ValueError("Checkpoint pointer read-back mismatch")
        checked_load(path, checksum)
        return current

    def resume(self, requested, bindings):
        pointer = json.loads(self.pointer.read_text())
        current = pointer["current"]
        exact = (self.run / current["path"]).resolve()
        if Path(requested).resolve() != exact:
            raise ValueError("Explicit resume must name CURRENT checkpoint; no older-checkpoint fallback")
        if json.loads(exact.with_suffix(".json").read_text()) != current:
            raise ValueError("Checkpoint sidecar mismatch")
        status = json.loads((self.run / "STATUS.json").read_text())
        if status["status"] != "PAUSED" or status.get("checkpoint") != current:
            raise ValueError("Only a durably acknowledged cooperative pause can resume; inspect unclean interruption separately")
        if pointer["previous_good"]:
            previous = pointer["previous_good"]
            checked_load(self.run / previous["path"], previous["sha256"])
        state = checked_load(exact, current["sha256"])
        if state["bindings"] != bindings:
            raise ValueError("Code/config/input/environment binding changed; refusing resume")
        if state["phase"] == "complete":
            raise ValueError("Completed results are immutable")
        return state


def update_status(run, status, state, **extra):
    atomic_json(run / "STATUS.json", {"status": status, "pid": os.getpid(),
                "phase": state["phase"], "update": state["update"],
                "epoch": state["epoch"], "batch_cursor": state["batch_cursor"],
                "prediction_cursor": state["prediction_cursor"],
                "budget_total_updates": state["budget_total_updates"], **extra})


def metric_summary(prediction, target, donors=None):
    p = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    pc, yc = p - p.mean(axis=0), y - y.mean(axis=0)
    denominator = np.sqrt((pc * pc).sum(axis=0) * (yc * yc).sum(axis=0))
    valid = denominator > 0
    correlations = np.full(y.shape[1], np.nan)
    correlations[valid] = (pc[:, valid] * yc[:, valid]).sum(axis=0) / denominator[valid]
    result = {"mse": float(np.mean((p - y) ** 2)), "n_rows": len(y),
            "pcc_mean_nonconstant": float(correlations[valid].mean()) if valid.any() else None,
            "pcc_valid_genes": int(valid.sum()), "pcc_total_genes": y.shape[1],
            "gene_pcc": [float(v) if np.isfinite(v) else None for v in correlations]}
    if donors is not None:
        result['per_donor']={str(d):metric_summary(p[donors==d],y[donors==d]) for d in np.unique(donors)}
        result['mean_donor_mse']=float(np.mean([r['mse'] for r in result['per_donor'].values()]))
        pcs=[r['pcc_mean_nonconstant'] for r in result['per_donor'].values() if r['pcc_mean_nonconstant'] is not None]
        result['mean_donor_pcc']=float(np.mean(pcs)) if pcs else None
    return result


def handler(_signal, _frame):
    global SIGNAL_PAUSE
    SIGNAL_PAUSE = True


def run_training(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    data, input_hash = load_data(args.input)
    n, d = data["X"].shape
    g = data["Y"].shape[1]
    batches = math.ceil(len(data["train_idx"]) / args.batch_size)
    config = {"seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
              "learning_rate": args.lr, "weight_decay": args.weight_decay,
              "hidden_dim": 128, "dropout": 0.1, "activation": "GELU",
              "scheduler": "StepLR", "scheduler_step_size": 1, "scheduler_gamma": 0.9,
              "loss": "MSE", "feature_scaling": "train_mean_std_ddof0_floor_1e-8",
              "target_transform": "as_supplied_no_fit", "checkpoint_every_updates": args.checkpoint_every,
              "prediction_batch_size": args.batch_size, "scope": "development_pilot_not_independent_generalization",
              "best_epoch_selection":"minimum_validation_mean_donor_MSE"}
    bindings = {"code_sha256": digest(Path(__file__).resolve()),
                "config_sha256": hashlib.sha256(canonical(config).encode()).hexdigest(),
                "config": config, "input_sha256": input_hash, "input_path": str(Path(args.input).resolve()),
                "environment": version_binding()}
    if args.pause_after_updates is not None and not 0 < args.pause_after_updates <= args.epochs * batches:
        raise ValueError("pause-after-updates must be within the fixed total update budget")
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock_file = open(run / ".run.lock", "a+")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        raise RuntimeError("Run is exclusively locked by another process")
    with lock_file:
        if (run / "results").exists():
            raise FileExistsError("Completed result path exists; no overwrite or retry")
        nonlock = [p for p in run.iterdir() if p.name != ".run.lock"]
        if not args.resume and nonlock:
            raise FileExistsError("New run requires an empty directory; existing state requires explicit resume")
        checkpoints = Checkpoints(run)
        random.seed(args.seed)
        np.random.seed(args.seed % (2**32))
        torch.manual_seed(args.seed)
        model = Head(d, g).cpu()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay, foreach=False, fused=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        if args.resume:
            if Path(args.pause_file).exists():
                raise ValueError("Pause request still exists; explicitly clear it before resuming")
            state = checkpoints.resume(args.resume, bindings)
            if args.pause_after_updates is not None and args.pause_after_updates <= state["update"]:
                raise ValueError("Requested pause update has already been consumed")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            restore_rng(state["rng"])
        else:
            mean = data["X"][data["train_idx"]].astype(np.float64).mean(axis=0)
            std = data["X"][data["train_idx"]].astype(np.float64).std(axis=0)
            std = np.maximum(std, 1e-8)
            sampler = torch.Generator(device="cpu").manual_seed(args.seed + 104729)
            plan = [torch.tensor(data["train_idx"])[torch.randperm(len(data["train_idx"]), generator=sampler)]
                    for _ in range(args.epochs)]
            state = {"schema_version": SCHEMA_VERSION, "bindings": bindings,
                     "phase": "train", "epoch": 0, "update": 0, "batch_cursor": 0,
                     "budget_total_updates": args.epochs * batches, "batches_per_epoch": batches,
                     "epoch_loss_sum": 0.0, "epoch_rows_seen": 0,
                     "history": [], "best_val_mse": None, "best_epoch": None, "best_model": None,
                     "sample_plan": plan, "sampler_generator": sampler.get_state(),
                     "loader_generator_before": None, "loader_generator_after": None,
                     "feature_mean": torch.tensor(mean, dtype=torch.float64),
                     "feature_std": torch.tensor(std, dtype=torch.float64),
                     "prediction_cursor": 0, "predictions": torch.zeros((n, g), dtype=torch.float32),
                     "training_final_model": None, "accumulation_position": 0, "gradients": None,
                     "applicability": {"accelerator": "none_CPU_only", "AMP_scaler": "not_used",
                                       "EMA_teacher": "not_used", "DDP": "not_supported",
                                       "worker_rng": "no_workers", "augmentation": "none_cached_features"}}
            atomic_json(run / "CONFIG.json", bindings, immutable=True)
        x_scaled = ((data["X"].astype(np.float64) - state["feature_mean"].numpy()) /
                    state["feature_std"].numpy()).astype(np.float32)
        if not np.isfinite(x_scaled).all():
            raise ValueError("Non-finite scaled features")
        x, y = torch.from_numpy(x_scaled), torch.from_numpy(data["Y"])
        dataset = TensorDataset(x, y)
        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

        def snapshot():
            state["model"] = copy.deepcopy(model.state_dict())
            state["optimizer"] = copy.deepcopy(optimizer.state_dict())
            state["scheduler"] = copy.deepcopy(scheduler.state_dict())
            state["rng"] = rng_state()
            if any(p.grad is not None for p in model.parameters()):
                raise ValueError("Checkpoint boundary retains gradients")
            return checkpoints.save(state)

        def pause_requested():
            return SIGNAL_PAUSE or Path(args.pause_file).exists() or (
                args.pause_after_updates is not None and state["update"] >= args.pause_after_updates)

        def pause():
            ref = snapshot()
            update_status(run, "PAUSED", state, checkpoint=ref,
                          acknowledgement="durable_checkpoint_verified_no_workers_process_will_exit")
            print(canonical({"status": "PAUSED", "checkpoint": str(run / ref["path"]),
                             "update": state["update"], "exit_code": PAUSE_EXIT}), flush=True)
            return PAUSE_EXIT

        optimizer.zero_grad(set_to_none=True)
        if not args.resume:
            snapshot()  # previous-good initial state precedes any updates
        update_status(run, "RUNNING", state)
        if pause_requested():
            return pause()
        while state["phase"] == "train":
            epoch = state["epoch"]
            plan = state["sample_plan"][epoch]
            cursor = state["batch_cursor"]
            generator = torch.Generator(device="cpu").manual_seed(args.seed + 15485863 + epoch)
            before = generator.get_state()
            batch_plan = [plan[i:i + args.batch_size].tolist()
                          for i in range(cursor * args.batch_size, len(plan), args.batch_size)]
            loader = DataLoader(dataset, batch_sampler=batch_plan, num_workers=0,
                                generator=generator, pin_memory=False)
            iterator = iter(loader)
            after = generator.get_state()
            if cursor and (not torch.equal(before, state["loader_generator_before"]) or
                           not torch.equal(after, state["loader_generator_after"])):
                raise ValueError("DataLoader iterator generator replay mismatch")
            state["loader_generator_before"], state["loader_generator_after"] = before, after
            for xb, yb in iterator:
                update_status(run, "UPDATE_IN_PROGRESS", state)
                model.train()
                loss = torch.mean((model(xb) - yb) ** 2)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise ValueError("Non-finite gradient")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                state["update"] += 1
                state["batch_cursor"] += 1
                state["epoch_loss_sum"] += float(loss.detach()) * len(xb)
                state["epoch_rows_seen"] += len(xb)
                finite_tree(model.state_dict(), "model")
                finite_tree(optimizer.state_dict(), "optimizer")
                epoch_finished = state["batch_cursor"] == batches
                if epoch_finished:
                    model.eval()
                    val_indices = data["val_idx"]
                    with torch.inference_mode():
                        val_pred = torch.cat([model(x[part]) for part in
                                              np.array_split(val_indices, max(1, math.ceil(len(val_indices) / args.batch_size)))])
                    val_metrics = metric_summary(val_pred.numpy(), data["Y"][val_indices], data['donors'][val_indices])
                    val_loss = val_metrics["mean_donor_mse"]
                    if state["best_val_mse"] is None or val_loss < state["best_val_mse"]:
                        state["best_val_mse"], state["best_epoch"] = val_loss, epoch + 1
                        state["best_model"] = copy.deepcopy(model.state_dict())
                    state["history"].append({"epoch": epoch + 1, "update": state["update"],
                        "train_mse": state["epoch_loss_sum"] / state["epoch_rows_seen"],
                        "validation": val_metrics, "learning_rate": optimizer.param_groups[0]["lr"]})
                    scheduler.step()
                    state["epoch"] += 1
                    state["batch_cursor"] = 0
                    state["epoch_loss_sum"], state["epoch_rows_seen"] = 0.0, 0
                    state["loader_generator_before"] = state["loader_generator_after"] = None
                    if state["epoch"] == args.epochs:
                        if state["update"] != state["budget_total_updates"]:
                            raise ValueError("Consumed update budget mismatch")
                        state["training_final_model"] = copy.deepcopy(model.state_dict())
                        model.load_state_dict(state["best_model"])
                        state["phase"] = "predict"
                update_status(run, "RUNNING", state)
                if pause_requested():
                    return pause()
                if epoch_finished or state["update"] % args.checkpoint_every == 0:
                    snapshot()
        model.eval()
        while state["prediction_cursor"] < n:
            start = state["prediction_cursor"]
            end = min(n, start + args.batch_size)
            update_status(run, "PREDICTION_IN_PROGRESS", state)
            with torch.inference_mode():
                prediction = model(x[start:end])
            finite_tree(prediction, "prediction")
            state["predictions"][start:end] = prediction
            state["prediction_cursor"] = end
            update_status(run, "RUNNING", state)
            if pause_requested():
                return pause()
        if digest(args.input) != input_hash or digest(__file__) != bindings["code_sha256"]:
            raise ValueError("Input or code changed during run")
        state["phase"] = "complete"
        final_ref = snapshot()
        predictions = state["predictions"].numpy()
        metrics = {"scope": config["scope"], "best_epoch": state["best_epoch"],
                   "train": metric_summary(predictions[data["train_idx"]], data["Y"][data["train_idx"]], data['donors'][data['train_idx']]),
                   "validation": metric_summary(predictions[data["val_idx"]], data["Y"][data["val_idx"]], data['donors'][data['val_idx']]),
                   "total_updates": state["update"], "history": state["history"]}
        staging = run / (".results-" + uuid.uuid4().hex)
        staging.mkdir()
        npz = staging / "predictions.npz"
        with open(npz, "xb") as f:
            np.savez(f, predictions=predictions, row_ids=data["row_ids"],
                     gene_names=data["gene_names"], coords=data["coords"],
                     train_idx=data["train_idx"], val_idx=data["val_idx"])
            f.flush()
            os.fsync(f.fileno())
        with np.load(npz, allow_pickle=False) as reread:
            if not np.array_equal(reread["predictions"], predictions):
                raise ValueError("Prediction read-back failed")
        atomic_json(staging / "metrics.json", metrics, immutable=True)
        atomic_json(staging / "manifest.json", {"predictions_sha256": digest(npz),
                    "metrics_sha256": digest(staging / "metrics.json"),
                    "final_checkpoint": final_ref, "bindings": bindings}, immutable=True)
        fsync_dir(staging)
        if (run / "results").exists():
            raise FileExistsError("Immutable results already exist")
        os.rename(staging, run / "results")
        fsync_dir(run)
        update_status(run, "COMPLETED", state, checkpoint=final_ref)
        print(canonical({"status": "COMPLETED", "results": str(run / "results"),
                         "updates": state["update"]}), flush=True)
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", metavar="EXACT_CURRENT_CHECKPOINT")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--pause-after-updates", type=int)
    parser.add_argument("--pause-file", default=None)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.checkpoint_every) < 1 or args.lr <= 0 or args.weight_decay < 0:
        parser.error("Invalid positive budget/batch/checkpoint/optimizer settings")
    if args.seed < 0 or args.seed > 2**63 - 15485863 - args.epochs:
        parser.error("Seed outside supported range")
    if not math.isfinite(args.lr) or not math.isfinite(args.weight_decay):
        parser.error("Optimizer settings must be finite")
    args.pause_file = args.pause_file or str(Path(args.run_dir) / "PAUSE_REQUEST")
    started = time.monotonic()
    try:
        code = run_training(args)
    except Exception as error:
        print(canonical({"status": "FAILED_NO_RETRY", "type": type(error).__name__,
                         "message": str(error), "elapsed_seconds": time.monotonic() - started}),
              file=sys.stderr, flush=True)
        return 1
    print(canonical({"elapsed_seconds": time.monotonic() - started, "exit_code": code}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

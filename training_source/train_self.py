#!/usr/bin/env python3
"""CPU source-adapted DeepSpot with explicit durable pause/exit/resume.

New standard-target arm: train-only sklearn scaler, raw-unit validation selection.
No COAST imports. Frozen train_head persistence is reused without modification.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import signal
import sys
import time
import uuid
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'03_code_environment/src'))
import train_head as persistence
import deepspot_target_scaler_v1 as target_scaling
from deepspot_adapter import DeepSpot, UPSTREAM_COMMIT, PUBLICATION_PIPELINE_COMMIT

BASE = Path(__file__).resolve().parents[2]/'03_code_environment'
ENGINEERING = 'engineering_native_p2_continuation_not_patient_validation'
RESEARCH = 'donor_separated_development'
PAUSE_EXIT = 75
PAUSE_REQUESTED = False
MEMORY_CAP = 16 * 1024**3
DISK_FLOOR = 100 * 1024**3
FROZEN_PERSISTENCE_SHA = '31d5ec723f24a6eec3ec7fe1dbddfd982696a6309c9b443d51d07d9c4d30fc9f'

def require(value, message):
    if not value:
        raise ValueError(message)

def on_pause(_signum, _frame):
    global PAUSE_REQUESTED
    PAUSE_REQUESTED = True

def npz(path, required):
    before = persistence.digest(path)
    with np.load(path, allow_pickle=False) as archive:
        require(set(required).issubset(archive.files), 'Missing NPZ fields: ' + str(required))
        result = {k: archive[k].copy() for k in archive.files}
    require(persistence.digest(path) == before, 'NPZ changed while reading')
    return result

def text_vector(value, length, name, unique=False):
    require(value.shape == (length,) and value.dtype.kind in 'US', name + ': invalid string array')
    require(not np.any(value == ''), name + ': empty identifier')
    if unique:
        require(len(np.unique(value)) == length, name + ': duplicate identifier')

def indices(value, n, name, nonempty=True):
    require(value.ndim == 1 and value.dtype.kind in 'iu', name + ': expected integer vector')
    require((not nonempty or len(value) > 0) and np.all(value >= 0) and np.all(value < n), name + ': empty or out of range')
    require(len(np.unique(value)) == len(value), name + ': duplicate index')
    return value.astype(np.int64)

def input_paths(args):
    names = ('features','rows','targets','split','context_cache','context_provenance','feature_provenance')
    return {name: Path(getattr(args, name)).resolve() for name in names}

def file_bindings(paths):
    return {key: {'path': str(path), 'sha256': persistence.digest(path), 'bytes': path.stat().st_size}
            for key, path in paths.items()}

def sources(builder):
    require(persistence.digest(BASE/'src/train_head.py') == FROZEN_PERSISTENCE_SHA,
            'Frozen shared persistence changed; new review required')
    paths = [Path(__file__).resolve(), BASE/'src/train_head.py', BASE/'src/deepspot_adapter.py',
        BASE/'third_party/DeepSpot/model.py', BASE/'third_party/DeepSpot/loss.py',
        BASE/'third_party/DeepSpot/utils.py', BASE/'third_party/DeepSpot/SOURCE.json', Path(builder).resolve(), *target_scaling.source_paths(), Path(__file__).resolve().parents[2]/'V2_BIOLOGICAL_REDESIGN/contracts/DEEPSPOT_CONTEXT_ATTRIBUTION_CONTROL.md']
    return {str(path): persistence.digest(path) for path in paths}

def read_data(args):
    paths = input_paths(args)
    bound = file_bindings(paths)
    features = np.load(paths['features'], mmap_mode='r', allow_pickle=False)
    require(features.ndim == 3 and features.shape[1:] == (10,768) and features.dtype == np.float32,
            'Require real raw float32 features[N,10,768]')
    n = len(features)
    require(n > 0, 'Empty native feature cache')
    rows = npz(paths['rows'], ('row_ids','samples','donors','coords','grid_coords'))
    # The predictor row file must be geometry and identifiers only.
    allowed = {'row_ids','samples','donors','coords','grid_coords'}
    require(set(rows).issubset(allowed), 'Unexpected field in image-only row file')
    for key in ('row_ids','samples','donors'):
        text_vector(rows[key], n, key, unique=key=='row_ids')
    require(rows['coords'].shape == (n,2) and rows['coords'].dtype.kind == 'f'
            and np.isfinite(rows['coords']).all(), 'Finite physical coords[N,2] required')
    require(rows['grid_coords'].shape == (n,2) and rows['grid_coords'].dtype.kind in 'iu',
            'Integer array/grid coordinates required')
    for sample in np.unique(rows['samples']):
        ix = np.flatnonzero(rows['samples'] == sample)
        require(len(np.unique(rows['donors'][ix])) == 1, 'One sample contains multiple donor IDs')
        require(len(np.unique(rows['grid_coords'][ix], axis=0)) == len(ix), 'Duplicate sample/grid coordinate')
    split = npz(paths['split'], ('train_idx','val_idx'))
    require(set(split).issubset({'train_idx','val_idx','context_only_idx'}), 'Unknown split role')
    train = indices(split['train_idx'], n, 'train_idx')
    val = indices(split['val_idx'], n, 'val_idx')
    require(not np.intersect1d(train, val).size, 'Training/validation query rows overlap')
    # Keep loader roles contiguous so final predictions use the same padding
    # width as each original train/validation loader, even when their maxima differ.
    query = np.concatenate([train, val])
    unused = np.setdiff1d(np.arange(n), query)
    if 'context_only_idx' in split:
        context_only = indices(split['context_only_idx'], n, 'context_only_idx', nonempty=False)
        require(np.array_equal(np.sort(context_only), unused), 'Context-only roles do not exactly cover unused rows')
    else:
        require(args.scope == ENGINEERING or not len(unused), 'Research unused rows require explicit context_only_idx')
        context_only = unused
    overlap = set(rows['donors'][train]) & set(rows['donors'][val])
    if args.scope == ENGINEERING:
        require(set(rows['donors']) == {'Oliveira2025_P2'}, 'Engineering exception is only the unchanged exposed P2 donor')
        require(all(str(x).startswith('GSM8594568:') for x in rows['row_ids']), 'Engineering input is not the approved P2 row namespace')
        require(len(train) >= 2048 and len(val) >= 1024 and args.batch_size == 1024 and args.epochs == 2,
                'Native qualification requires >=2048/1024 real query rows, batch1024, two epochs')
    else:
        require(not overlap, 'Research mode rejects ANY training/validation donor overlap')
        require(set(rows['donors'][context_only]).issubset(set(rows['donors'][query])),
                'Context-only donor must have an explicitly assigned train or validation query role')

    targets = npz(paths['targets'], ('Y','row_ids','gene_names'))
    require(set(targets).issubset({'Y','row_ids','gene_names','gene_ids'}), 'Unknown target file field')
    require(np.array_equal(targets['row_ids'], rows['row_ids']), 'Targets and image rows misaligned')
    y = targets['Y']
    require(y.ndim == 2 and y.shape[0] == n and y.shape[1] > 0 and y.dtype == np.float32,
            'Require common-unit float32 Y[N,G]')
    require(np.isfinite(y[query]).all(), 'Nonfinite query targets')
    text_vector(targets['gene_names'], y.shape[1], 'gene_names', unique=True)
    if 'gene_ids' in targets:
        text_vector(targets['gene_ids'], y.shape[1], 'gene_ids', unique=True)
    # Context-only Y values never enter model input, loss, eligibility or metrics.

    feature_meta = json.loads(paths['feature_provenance'].read_text())
    require(feature_meta.get('features_sha256') == bound['features']['sha256']
            and feature_meta.get('rows_sha256') == bound['rows']['sha256'], 'Feature/row provenance hashes disagree')
    require(feature_meta.get('n_views') == 10 and feature_meta.get('native_subviews') is True
            and feature_meta.get('subviews_source') == 'native_before_resize', 'True native nine-subview provenance required')
    cache = npz(paths['context_cache'], ('row_ids','indptr','indices'))
    require(set(cache) == {'row_ids','indptr','indices'}, 'CSR cache must have geometry fields only')
    require(np.array_equal(cache['row_ids'], rows['row_ids']), 'Context rows misaligned')
    offsets, neighbors = cache['indptr'], cache['indices']
    require(offsets.shape == (n+1,) and offsets.dtype.kind in 'iu' and neighbors.ndim == 1
            and neighbors.dtype.kind in 'iu', 'Malformed CSR arrays')
    require(offsets[0] == 0 and offsets[-1] == len(neighbors) and np.all(offsets[1:] >= offsets[:-1]), 'Invalid CSR offsets')
    require(np.all(neighbors >= 0) and np.all(neighbors < n), 'Context row out of range')
    # Independent geometry reconstruction: source axis-aligned radius1, original
    # row order, same sample, self excluded. Targets are not consulted here.
    lookup = {(str(s),int(r),int(c)): i for i,(s,(r,c)) in enumerate(zip(rows['samples'],rows['grid_coords']))}
    for i,(sample,(r,c)) in enumerate(zip(rows['samples'],rows['grid_coords'])):
        expected = sorted(lookup[key] for dr in (-1,0,1) for dc in (-1,0,1)
                          if (dr or dc) and (key := (str(sample),int(r)+dr,int(c)+dc)) in lookup)
        actual = neighbors[offsets[i]:offsets[i+1]]
        require(np.array_equal(actual, np.asarray(expected,dtype=np.int64)), 'CSR differs from image-only radius1 rule at row '+str(i))
    context_meta = json.loads(paths['context_provenance'].read_text())
    require(context_meta.get('rows_sha256') == bound['rows']['sha256']
            and context_meta.get('context_sha256') == bound['context_cache']['sha256'], 'Context provenance hash mismatch')
    builder = Path(context_meta['builder_path']).resolve()
    require(builder.is_file() and persistence.digest(builder) == context_meta.get('builder_sha256'), 'Context builder hash mismatch')
    require(context_meta.get('geometry_uses_expression') is False
            and context_meta.get('neighbor_view') == 'slot0_only' and context_meta.get('neighbor_radius') == 1,
            'Expression-free slot0-only source neighbor declaration required')
    require(isinstance(context_meta.get('geometry_profile'), str) and bool(context_meta['geometry_profile'].strip()),
            'Explicit geometry adaptation/profile required')
    degree = np.diff(offsets)
    pads = {name: max(1,int(degree[ix].max())) for name,ix in [('train',train),('validation',val),('prediction',query)]}
    require(file_bindings(paths) == bound, 'Input changed during read/validation')
    return {'features':features, 'Y':y, 'rows':rows, 'genes':targets['gene_names'], 'gene_ids':targets.get('gene_ids'),
        'train':train, 'val':val, 'query':query, 'context_only':context_only,
        'indptr':offsets, 'neighbors':neighbors, 'pads':pads,
        'geometry_profile':context_meta['geometry_profile'], 'builder':builder, 'paths':paths, 'bound':bound}

def image_batch(features, query_rows, indptr, neighbor_indices, padding_width):
    """True main+9 target views; repeated query SLOT0 in real neighbor slots; no targets."""
    views = np.asarray(features[query_rows], dtype=np.float32)
    neighbor_values = np.zeros((len(query_rows),padding_width,768),dtype=np.float32)
    for b,i in enumerate(query_rows):
        selected = neighbor_indices[indptr[i]:indptr[i+1]]
        require(len(selected) <= padding_width, 'Loader padding smaller than true neighborhood')
        if len(selected):
            neighbor_values[b,:len(selected)] = features[i,0,:]  # V2 matched control: all real slots receive self
    require(np.isfinite(views).all() and np.isfinite(neighbor_values).all(), 'Nonfinite image-only input')
    # Contiguous copies are essential because the unchanged source uses view().
    return [torch.from_numpy(np.ascontiguousarray(views[:,:1,:])),
            torch.from_numpy(np.ascontiguousarray(views[:,1:,:])), torch.from_numpy(neighbor_values)]

def prediction_batch(model, features, query_rows, indptr, neighbor_indices, padding_width):
    """Predictor interface deliberately has no Y, scaler or library-size input."""
    model.eval()
    with torch.inference_mode():
        return model(image_batch(features,query_rows,indptr,neighbor_indices,padding_width))

def metrics(prediction, targets, donors):
    p=np.asarray(prediction,dtype=np.float64);y=np.asarray(targets,dtype=np.float64)
    result={}
    for donor in np.unique(donors):
        chosen=donors==donor;a=p[chosen];b=y[chosen]
        good=(np.ptp(a,axis=0)>1e-12*np.maximum(1,np.abs(a).max(0)))&(np.ptp(b,axis=0)>1e-12*np.maximum(1,np.abs(b).max(0)))
        ac=a-a.mean(0);bc=b-b.mean(0);den=np.sqrt((ac*ac).sum(0)*(bc*bc).sum(0))
        corr=np.full(y.shape[1],np.nan);corr[good]=(ac*bc).sum(0)[good]/den[good]
        result[str(donor)]={'query_rows':int(chosen.sum()),'mse':float(np.mean((a-b)**2)),
            'mean_gene_pcc':float(corr[good].mean()) if good.any() else None,
            'n_defined_genes':int(good.sum()),'gene_pcc':[float(x) if np.isfinite(x) else None for x in corr]}
    cor=[v['mean_gene_pcc'] for v in result.values() if v['mean_gene_pcc'] is not None]
    return {'mean_donor_mse':float(np.mean([v['mse'] for v in result.values()])),
        'mean_donor_pcc':float(np.mean(cor)) if cor else None, 'per_donor':result}

def resource_estimate(data, args):
    n,g=data['Y'].shape;batch=args.batch_size;pad=max(data['pads'].values())
    params=10*(768*512+512)+10*(1536*g+g);weights=params*4
    stack=batch*(10+pad)*10*512*4
    inputs=batch*(10+pad)*768*4
    # Conservative accounting, not a measured activation peak. Eight copies of
    # the Phi stack bound stored activations, dropout masks and backward work.
    estimate=2*1024**3 + data['bound']['features']['bytes'] + 3*data['Y'].nbytes + weights*10 + stack*8 + inputs*4
    checkpoint=weights*6+4*(len(data['val'])+len(data['query']))*g+128*1024**2
    updates=math.ceil(len(data['train'])/batch)*args.epochs
    generations=1+args.epochs+updates//args.checkpoint_every+8
    storage=checkpoint*generations+8*len(data['train'])*args.epochs+8*n*g+16*1024**2
    return {'memory_cap_bytes':MEMORY_CAP,'conservative_runner_rss_estimate_bytes':int(estimate),
        'reserve_for_harness_and_estimate_error_bytes':4*1024**3,'phi_stacked_activation_bytes':int(stack),
        'batch_input_bytes':int(inputs),'parameters':params,'fp32_parameter_bytes':weights,
        'checkpoint_upper_estimate_bytes':int(checkpoint),'retained_generations_estimate':generations,
        'estimated_run_storage_bytes':int(storage),'disk_floor_bytes':DISK_FLOOR,
        'batch_size':batch,'max_padding_width':pad,'threads':2,'workers':0,'accelerator':'none',
        'runtime_status':'Requires actual approved native batch measurement; no full-run runtime claim',
        'limits':'macOS peak RSS checks plus harness combined-RSS watchdog; not an OS-enforced virtual-memory limit'}

def save_array(path, **arrays):
    path=Path(path)
    require(not path.exists(), 'Completed array exists')
    temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    with temp.open('xb') as f:
        np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
    with np.load(temp,allow_pickle=False) as check:
        require(set(check.files)==set(arrays) and all(np.array_equal(check[k],v) for k,v in arrays.items()), 'Array readback mismatch')
    checksum=persistence.digest(temp);os.rename(temp,path);persistence.fsync_dir(path.parent)
    require(persistence.digest(path)==checksum,'Published array checksum mismatch')
    return checksum

class Checkpoints(persistence.Checkpoints):
    def save(self,state):
        if self.pointer.exists():
            ptr=json.loads(self.pointer.read_text());ref=ptr['current']
            prior=persistence.checked_load(self.run/ref['path'],ref['sha256'])
            require(prior['bindings']==state['bindings'] and prior['update']<=state['update'], 'Previous-good binding/cursor mismatch')
        return super().save(state)
    def resume(self,requested,bindings):
        state=super().resume(requested,bindings)
        ptr=json.loads(self.pointer.read_text())
        require(state['update']==0 or ptr['previous_good'] is not None,'Missing previous good generation')
        if ptr['previous_good']:
            ref=ptr['previous_good'];prior=persistence.checked_load(self.run/ref['path'],ref['sha256'])
            require(prior['bindings']==bindings and prior['update']<=state['update'],'Invalid previous good generation')
        return state

def fitted_target_scaler(data,args):
    if not args.resume:
        return target_scaling.fit_training(data['Y'],data['train'])
    require(args.run_dir is not None,'Resume requires source run directory')
    run=Path(args.run_dir).resolve();pointer=json.loads((run/'CURRENT.json').read_text())
    ref=pointer['current'];current=(run/ref['path']).resolve()
    require(current.is_relative_to(run/'checkpoints') and Path(args.resume).resolve()==current,'Scaler resume requires exact CURRENT checkpoint')
    require(json.loads((run/'STATUS.json').read_text())['status']=='PAUSED','Scaler resume requires acknowledged PAUSED state')
    prior=persistence.checked_load(current,ref['sha256'])
    require(prior['bindings']['inputs']==data['bound'],'Scaler input bindings changed')
    scaler=copy.deepcopy(prior['target_scaler']);del prior
    target_scaling.verify_training_binding(scaler,data['Y'],data['train'])
    return scaler

def execute(args):
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    data=read_data(args);train=data['train'];val=data['val'];query=data['query'];n,g=data['Y'].shape
    batches=math.ceil(len(train)/args.batch_size);estimate=resource_estimate(data,args)
    target_scaler=fitted_target_scaler(data,args);scaler_sha=target_scaling.state_sha(target_scaler)
    require(estimate['conservative_runner_rss_estimate_bytes'] < MEMORY_CAP-4*1024**3,
            'Conservative native batch estimate breaches runner allowance; do not launch')
    config={'scope':args.scope,'donor_separation_required':args.scope!=ENGINEERING,'epochs':args.epochs,'batch_size':args.batch_size,
        'seed':args.seed,'learning_rate':args.lr,'optimizer':'AdamW','weight_decay':1e-6,
        'foreach':False,'fused':False,'optimizer_backend':'explicit CPU scalar path; same source AdamW equations',
        'loss':'MSE_mean_standardized_targets','scheduler':None,'resampling_resolution':0,'target_scaler':target_scaling.SCALER_NAME,'target_smoothing':0,
        'scaler_fit':'training eligible query rows only; population variance; upstream sklearn constant-feature rule',
        'prediction_units':'inverse transformed once to original target units; scaler never refit on resume/prediction',
        'features':'raw_frozen_Phikon_native_ten_views','target_transform':args.target_transform,
        'architecture':'ten_Phi_lower_median_target_subspot_sum_neighbor_max_ten_Rho_mean',
        'neighbor_views':'slot0_only','context_control':'all_real_neighbor_slots_replaced_by_query_slot0','neighbors_may_be_context_only_rows':False,
        'neighbor_Y_never_read':True,'geometry_profile':data['geometry_profile'],'padding_widths':data['pads'],
        'prediction_padding':'retain source train/validation loader width for the corresponding query rows',
        'best_epoch_selection':'minimum_validation_mean_donor_MSE_development_adaptation',
        'checkpoint_every':args.checkpoint_every,'upstream_commit':UPSTREAM_COMMIT,
        'publication_pipeline_commit':PUBLICATION_PIPELINE_COMMIT}
    code=sources(data['builder'])
    bindings={'inputs':data['bound'],'source_code_sha256':code,'config':config,
        'config_sha256':hashlib.sha256(persistence.canonical(config).encode()).hexdigest(),
        'environment':persistence.version_binding(),'target_scaler_sha256':scaler_sha,
        'target_scaler_sklearn_version':target_scaler['sklearn_version']}
    if args.preflight_only:
        require(args.preflight_output is not None,'Preflight requires a new explicit output file')
        persistence.atomic_json(Path(args.preflight_output),{'status':'PREFLIGHT_NO_MODEL_OR_TRAINING',
            'bindings':bindings,'resources':estimate,'rows':n,'train_queries':len(train),'validation_queries':len(val),
            'context_only_rows':len(data['context_only']),'scaler':{'sha256':scaler_sha,'training_rows':target_scaler['n_samples_seen'],'zero_variance_genes':int((target_scaler['var']==0).sum())},'limits':'No actual batch runtime or peak activation measurement yet'},immutable=True)
        print(persistence.canonical({'status':'PREFLIGHT_ONLY','report':args.preflight_output}));return 0
    require(args.run_dir is not None, 'Training requires run directory')
    run=Path(args.run_dir).resolve();run.mkdir(parents=True,exist_ok=True)
    require(shutil.disk_usage(run).free-estimate['estimated_run_storage_bytes']>=DISK_FLOOR,'Retained generations would breach disk floor')
    with (run/'.run.lock').open('a+') as lock,(BASE/'deepspot_cpu.lock').open('a+') as device_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(device_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not (run/'results').exists(),'Completed results immutable')
        if not args.resume:
            require(not any(p.name!='.run.lock' and not p.name.startswith('._') for p in run.iterdir()),'New run requires empty directory; no retries')
        pause_file=Path(args.pause_file) if args.pause_file else run/'PAUSE_REQUEST'
        require(not args.resume or not pause_file.exists(),'Explicitly remove owned pause request before resume')
        checkpoints=Checkpoints(run)
        # The original constructor intentionally seeds globals. Restore saved RNG
        # only AFTER constructing and loading model+optimizer on explicit resume.
        model=DeepSpot(input_size=768,output_size=g,loss_func='mse',random_seed=args.seed,
                       lr=args.lr,weight_decay=1e-6,scaler=None,spot_context='spot_subspot_neighbors').cpu()
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-6,foreach=False,fused=False)
        if args.resume:
            state=checkpoints.resume(args.resume,bindings)
            require(persistence.digest(run/'sample_plan.npz')==state['sample_plan_sha256'],'Immutable sample plan changed')
            plan=npz(run/'sample_plan.npz',('order','generator_state'))
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);persistence.restore_rng(state['rng'])
        else:
            generator=torch.Generator(device='cpu').manual_seed(args.seed+104729)
            order=np.stack([train[torch.randperm(len(train),generator=generator).numpy()] for _ in range(args.epochs)])
            plan={'order':order,'generator_state':generator.get_state().numpy()}
            ph=save_array(run/'sample_plan.npz',**plan)
            state={'schema_version':persistence.SCHEMA_VERSION,'bindings':bindings,'phase':'train',
                'epoch':0,'update':0,'batch_cursor':0,'validation_cursor':0,'prediction_cursor':0,
                'validation_batches_total':0,'prediction_batches_total':0,'budget_total_updates':batches*args.epochs,
                'batches_per_epoch':batches,'sample_plan_sha256':ph,'sampler_generator':torch.from_numpy(plan['generator_state'].copy()),
                'target_scaler':copy.deepcopy(target_scaler),'history':[],'epoch_loss_sum':0.,'epoch_rows_seen':0,'best_val_mse':None,'best_epoch':None,
                'best_model':None,'training_final_model':None,'validation_predictions':torch.zeros((len(val),g)),
                'predictions':torch.zeros((len(query),g)),'query_rows':torch.from_numpy(query.copy()),
                'accumulation_position':0,'gradients':None,'scheduler':None,'scaler':None,'EMA_teacher':None,
                'applicability':{'CPU':True,'accelerator':None,'workers':0,'DDP':'unsupported','augmentation':'none',
                    'dropout':'default source probabilities; complete CPU RNG checkpointed',
                    'loader':'memory-mapped ten-view cache; deterministic CSR context and immutable epoch order',
                    'qualification_statement':'not patient validation' if args.scope==ENGINEERING else 'donor-separated development; not final test'}}
            persistence.atomic_json(run/'CONFIG.json',bindings,immutable=True)
            persistence.atomic_json(run/'RESOURCE_BUDGET.json',estimate,immutable=True)
        require(target_scaling.state_sha(state['target_scaler'])==scaler_sha,'Saved target scaler differs')
        require(plan['order'].shape==(args.epochs,len(train)),'Sample-plan shape mismatch')
        for order in plan['order']:
            require(np.array_equal(np.sort(order),np.sort(train)),'Sample plan not a training permutation')
        require(state['budget_total_updates']==batches*args.epochs and state['batches_per_epoch']==batches,'Budget changed')
        for arg,key in [('pause_after_updates','update'),('pause_after_validation_batches','validation_batches_total'),
                        ('pause_after_prediction_batches','prediction_batches_total')]:
            threshold=getattr(args,arg)
            require(threshold is None or threshold>state[key], 'Pause threshold already consumed: '+arg)
        require(args.pause_after_updates is None or args.pause_after_updates<=state['budget_total_updates'],'Pause exceeds budget')
        optimizer.zero_grad(set_to_none=True)
        signal.signal(signal.SIGINT,on_pause);signal.signal(signal.SIGTERM,on_pause)
        started=time.monotonic();initial_update=state['update'];timing={name:0. for name in ('train','validate','predict')}
        phase_batches={name:0 for name in timing}
        def status(value,**extra):
            persistence.update_status(run,value,state,validation_cursor=state['validation_cursor'],
                validation_batches_total=state['validation_batches_total'],prediction_batches_total=state['prediction_batches_total'],**extra)
        def requested():
            return PAUSE_REQUESTED or pause_file.exists() or any(getattr(args,a) is not None and state[k]>=getattr(args,a)
                for a,k in [('pause_after_updates','update'),('pause_after_validation_batches','validation_batches_total'),
                            ('pause_after_prediction_batches','prediction_batches_total')])
        def snapshot():
            require(shutil.disk_usage(run).free-estimate['checkpoint_upper_estimate_bytes']>=DISK_FLOOR,'Checkpoint would breach disk floor')
            require(sources(data['builder'])==code,'Live source changed')
            require(all(p.grad is None for p in model.parameters()),'Checkpoint with unfinished gradients')
            state['model']=copy.deepcopy(model.state_dict());state['optimizer']=copy.deepcopy(optimizer.state_dict());state['rng']=persistence.rng_state()
            return checkpoints.save(state)
        def receipt(value,ref):
            persistence.atomic_json(run/('PROCESS_'+str(os.getpid())+'.json'),{'status':value,'pid':os.getpid(),
                'seconds_after_setup':time.monotonic()-started,'updates_this_process':state['update']-initial_update,
                'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'rss_unit':'macOS_bytes',
                'phase_seconds':timing,'phase_batches':phase_batches,'checkpoint':ref,
                'workers':0,'accelerator':None,'training_scope':args.scope},immutable=True)
        def pause():
            ref=snapshot();status('PAUSED',checkpoint=ref,acknowledgement='durable verified checkpoint then process exit; zero workers')
            receipt('PAUSED',ref);print(persistence.canonical({'status':'PAUSED','phase':state['phase'],'checkpoint':str(run/ref['path'])}),flush=True)
            return PAUSE_EXIT
        if not args.resume:
            snapshot()
        status('RUNNING')
        while state['phase']!='complete':
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<MEMORY_CAP,'Peak process RSS exceeded 16 GiB')
            if requested():
                return pause()
            phase=state['phase'];t=time.monotonic()
            if phase=='train':
                start=state['batch_cursor']*args.batch_size;end=min(start+args.batch_size,len(train))
                ix=plan['order'][state['epoch'],start:end]
                status('UPDATE_IN_PROGRESS');model.train()
                inputs=image_batch(data['features'],ix,data['indptr'],data['neighbors'],data['pads']['train'])
                prediction=model(inputs);target=target_scaling.transform_training(data['Y'][ix],target_scaler)
                loss=model.loss_func(prediction,target);persistence.finite_tree(loss,'loss');loss.backward()
                for param in model.parameters():
                    if param.grad is not None:
                        require(bool(torch.isfinite(param.grad).all()),'Nonfinite gradient')
                optimizer.step();optimizer.zero_grad(set_to_none=True)
                persistence.finite_tree(model.state_dict());persistence.finite_tree(optimizer.state_dict())
                state['update']+=1;state['batch_cursor']+=1;state['epoch_loss_sum']+=float(loss.detach())*len(ix);state['epoch_rows_seen']+=len(ix)
                del inputs,prediction,target,loss
                if state['batch_cursor']==batches:
                    state['phase']='validate';state['validation_cursor']=0
                timing[phase]+=time.monotonic()-t;phase_batches[phase]+=1;status('RUNNING')
                if requested():
                    return pause()
                if state['update']%args.checkpoint_every==0:
                    snapshot()
            elif phase=='validate':
                start=state['validation_cursor'];end=min(start+args.batch_size,len(val));ix=val[start:end]
                status('VALIDATION_PREDICTION_IN_PROGRESS')
                pred=target_scaling.inverse_prediction(prediction_batch(model,data['features'],ix,data['indptr'],data['neighbors'],data['pads']['validation']),target_scaler)
                persistence.finite_tree(pred);state['validation_predictions'][start:end]=pred;state['validation_cursor']=end
                state['validation_batches_total']+=1
                if end==len(val):
                    score=metrics(state['validation_predictions'].numpy(),data['Y'][val],data['rows']['donors'][val]);value=score['mean_donor_mse']
                    if state['best_val_mse'] is None or value<state['best_val_mse']:
                        state['best_val_mse']=value;state['best_epoch']=state['epoch']+1;state['best_model']=copy.deepcopy(model.state_dict())
                    state['history'].append({'epoch':state['epoch']+1,'update':state['update'],
                        'training_standardized_MSE':state['epoch_loss_sum']/state['epoch_rows_seen'],'validation':score,
                        'learning_rate':optimizer.param_groups[0]['lr'],'not_patient_validation':args.scope==ENGINEERING})
                    state['epoch']+=1;state['batch_cursor']=0;state['epoch_loss_sum']=0.;state['epoch_rows_seen']=0
                    if state['epoch']==args.epochs:
                        require(state['update']==state['budget_total_updates'],'Total update budget mismatch')
                        state['training_final_model']=copy.deepcopy(model.state_dict());model.load_state_dict(state['best_model']);state['phase']='predict'
                    else:
                        state['phase']='train'
                    snapshot()
                timing[phase]+=time.monotonic()-t;phase_batches[phase]+=1;status('RUNNING')
                del pred
            elif phase=='predict':
                start=state['prediction_cursor']
                role_end=len(train) if start<len(train) else len(query)
                role='train' if start<len(train) else 'validation'
                end=min(start+args.batch_size,role_end);ix=query[start:end]
                status('PREDICTION_IN_PROGRESS')
                pred=target_scaling.inverse_prediction(prediction_batch(model,data['features'],ix,data['indptr'],data['neighbors'],data['pads'][role]),target_scaler)
                persistence.finite_tree(pred);state['predictions'][start:end]=pred;state['prediction_cursor']=end;state['prediction_batches_total']+=1
                timing[phase]+=time.monotonic()-t;phase_batches[phase]+=1;status('RUNNING')
                del pred
                if end==len(query):
                    state['phase']='complete'
            else:
                raise ValueError('Unknown continuation phase')
        require(file_bindings(data['paths'])==data['bound'],'Bound input changed during run')
        require(persistence.digest(run/'sample_plan.npz')==state['sample_plan_sha256'],'Sample plan changed during run')
        ref=snapshot();pred=state['predictions'].numpy();query_position={int(v):i for i,v in enumerate(query)}
        train_positions=np.array([query_position[int(v)] for v in train]);val_positions=np.array([query_position[int(v)] for v in val])
        result={'scope':args.scope,'not_patient_validation':args.scope==ENGINEERING,'best_epoch':state['best_epoch'],
            'history':state['history'],'total_updates':state['update'],
            'train':metrics(pred[train_positions],data['Y'][train],data['rows']['donors'][train]),
            'validation':metrics(pred[val_positions],data['Y'][val],data['rows']['donors'][val])}
        staging=run/('.results-'+uuid.uuid4().hex);staging.mkdir()
        prediction_arrays={'predictions':pred,'row_ids':data['rows']['row_ids'][query],
            'gene_names':data['genes'],'query_rows':query,'train_query_positions':train_positions,'val_query_positions':val_positions}
        if data['gene_ids'] is not None:
            prediction_arrays['gene_ids']=data['gene_ids']
        checksum=save_array(staging/'predictions.npz',**prediction_arrays)
        persistence.atomic_json(staging/'metrics.json',result,immutable=True)
        persistence.atomic_json(staging/'manifest.json',{'bindings':bindings,'predictions_sha256':checksum,
            'metrics_sha256':persistence.digest(staging/'metrics.json'),'final_checkpoint':ref},immutable=True)
        persistence.fsync_dir(staging);require(not(run/'results').exists(),'Completed results already exist')
        os.rename(staging,run/'results');persistence.fsync_dir(run);status('COMPLETED',checkpoint=ref);receipt('COMPLETED',ref)
        print(persistence.canonical({'status':'COMPLETED','scope':args.scope,'results':str(run/'results')}),flush=True)
        return 0

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('features','rows','targets','split','context-cache','context-provenance','feature-provenance'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--scope',choices=[ENGINEERING,RESEARCH],required=True)
    parser.add_argument('--target-transform',required=True,help='Frozen target-unit description; no fitting occurs here')
    parser.add_argument('--run-dir');parser.add_argument('--resume');parser.add_argument('--pause-file')
    parser.add_argument('--epochs',type=int,default=15);parser.add_argument('--batch-size',type=int,default=1024)
    parser.add_argument('--seed',type=int,default=20261017);parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--checkpoint-every',type=int,default=128)
    parser.add_argument('--pause-after-updates',type=int);parser.add_argument('--pause-after-validation-batches',type=int)
    parser.add_argument('--pause-after-prediction-batches',type=int)
    parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--preflight-output')
    args=parser.parse_args()
    if min(args.epochs,args.batch_size,args.checkpoint_every)<1 or not math.isfinite(args.lr) or args.lr<=0 or not 0<=args.seed<2**32:
        parser.error('Invalid immutable training budget, seed or optimizer option')
    if any(x is not None and x<1 for x in (args.pause_after_updates,args.pause_after_validation_batches,args.pause_after_prediction_batches)):
        parser.error('Pause thresholds must be positive')
    if not args.target_transform.strip():
        parser.error('Explicit target units required')
    try:
        return execute(args)
    except Exception as error:
        print(persistence.canonical({'status':'FAILED_NO_RETRY','type':type(error).__name__,'message':str(error)}),file=sys.stderr,flush=True)
        return 1

if __name__=='__main__':
    raise SystemExit(main())

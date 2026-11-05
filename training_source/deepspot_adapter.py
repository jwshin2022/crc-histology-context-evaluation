"""Pure-Torch DeepSpot adaptation, MIT; see third_party/DeepSpot/NOTICE.md.

Full ten-Phi/ten-Rho mathematical method bodies are source-identical ASTs.
Lightning trainer/logging is absent. Use forward([main, nine_subspots, neighbors])
and model.loss_func(prediction, target). This file is NOT a resumable runner.
Construction retains upstream global fix_seed behavior; restore RNG afterward
when resuming. All inputs must be raw, contiguous float32 encoder features.
"""
from typing import List, Tuple, Union
from enum import Enum
from types import SimpleNamespace
from pathlib import Path
import hashlib
import importlib.util
from torch.backends import cudnn
from torch import nn
import numpy as np
import random
import torch
import os

UPSTREAM_COMMIT = '90a2830cb72d2f41de16df2c0e3be71140fae29f'
PUBLICATION_PIPELINE_COMMIT = '17087753ce410a9fbd928255679c8c42c088bc58'
_VENDOR = Path(__file__).resolve().parents[1] / 'third_party/DeepSpot'
_SOURCE_HASHES = {'model.py': '6ecc31e43f0ce14541e6f4336d8cc3af2ac3a52f1afaa7e6f1eb67c43cfa2630', 'loss.py': 'e5801f8449c118b4af494d9f3f7d91f3b75b6c1f66380134a8c33b6b82aa9234', 'utils.py': '263d12ea61e4e455f26a65fbfbb9f35efff79d7b8f000d49e39cb228098f3942', 'LICENSE': '6f623e2c0cfa80c1bd047809e448f865a51566548f7afd7a6ed30cacaa678d93'}
for _name, _expected in _SOURCE_HASHES.items():
    if hashlib.sha256((_VENDOR / _name).read_bytes()).hexdigest() != _expected:
        raise RuntimeError('Pinned DeepSpot source changed: ' + _name)
_spec = importlib.util.spec_from_file_location('_deepspot_adapter_loss', _VENDOR / 'loss.py')
_loss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_loss)
loss_cosine_function = _loss.loss_cosine_function
loss_pearson_function = _loss.loss_pearson_function
loss_mse_function = _loss.loss_mse_function
loss_poisson_function = _loss.loss_poisson_function
loss_mse_pearson_function = _loss.loss_mse_pearson_function
loss_mse_cosine_function = _loss.loss_mse_cosine_function

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

class Operation(str, Enum):
    SUM = 'sum'
    MAX = 'max'
    NONE = 'none'
    MEAN = 'mean'

class Phi(nn.Module):

    def __init__(self, input_size: int, output_size: int, p: float=0.0):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_size, output_size), nn.Dropout(p=p), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class Rho(nn.Module):

    def __init__(self, input_size: int, output_size: int, p: float=0.0):
        super().__init__()
        self.model = nn.Sequential(nn.Dropout(p=p), nn.ReLU(inplace=True), nn.Linear(input_size, output_size), nn.Dropout(p=0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class DeepSpot(nn.Module):

    def __init__(self, input_size: int, output_size: int, loss_func: str='mse', lr: float=0.0001, p: float=0.3, p_phi: Union[None, float]=None, p_rho: Union[None, float]=None, n_ensemble: int=10, n_ensemble_phi: Union[None, int]=None, n_ensemble_rho: Union[None, int]=None, phi2rho_size: int=512, weight_decay: float=1e-06, random_seed: int=2024, scaler=None, spot_context: str='spot_subspot_neighbors'):
        super().__init__()
        self.hparams = SimpleNamespace(**{k: v for k, v in locals().items() if k not in ('self', '__class__')})
        self.scaler = scaler
        fix_seed(random_seed)
        if loss_func == 'mse':
            self.loss_func = loss_mse_function
        elif loss_func == 'cos':
            self.loss_func = loss_cosine_function
        elif loss_func == 'mse_cos':
            self.loss_func = loss_mse_cosine_function
        elif loss_func == 'pearson':
            self.loss_func = loss_pearson_function
        elif loss_func == 'mse_pearson':
            self.loss_func = loss_mse_pearson_function
        elif loss_func == 'poisson':
            self.loss_func = loss_poisson_function
        self.p_phi = p_phi or p
        self.p_rho = p_rho or p
        self.n_ensemble_phi = n_ensemble_phi or n_ensemble
        self.n_ensemble_rho = n_ensemble_rho or n_ensemble
        self.training_loss = []
        self.validation_loss = []
        self.phi2rho_size = phi2rho_size
        self.phi_spot = nn.ModuleList([Phi(input_size, phi2rho_size, self.p_phi) for _ in range(self.n_ensemble_phi)])
        self.rho = nn.ModuleList([Rho(phi2rho_size * self._get_phi_multiplier(spot_context), output_size, self.p_rho) for _ in range(self.n_ensemble_rho)])
        self._forward_fn = self._get_forward_function(spot_context)

    def _get_phi_multiplier(self, spot_context: str) -> int:
        context_multipliers = {'spot': 1, 'spot_subspot': 2, 'spot_neighbors': 2, 'spot_subspot_neighbors': 3}
        return context_multipliers.get(spot_context, 1)

    def _get_forward_function(self, spot_context: str):
        forward_functions = {'spot': self._forward_spot, 'spot_subspot': self._forward_spot_subspot, 'spot_neighbors': self._forward_spot_neighbors, 'spot_subspot_neighbors': self._forward_spot_subspot_neighbors}
        return forward_functions.get(spot_context, self._forward_spot)

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        x = self._forward_fn(x)
        return x

    def inverse_transform(self, x) -> torch.Tensor:
        if self.scaler is not None:
            x = self.scaler.inverse_transform(x)
        return x

    def _forward_spot(self, x: torch.Tensor) -> torch.Tensor:
        x_phi = self._apply_phi(x, self.phi_spot, operation=Operation.SUM)
        return self._apply_rho(x_phi)

    def _forward_spot_subspot(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_spot, x_subspot = x
        x_subspot = self._apply_phi(x_subspot, self.phi_spot, operation=Operation.SUM)
        x_spot = self._apply_phi(x_spot, self.phi_spot)
        x_phi = torch.cat((x_spot, x_subspot), dim=1)
        return self._apply_rho(x_phi)

    def _forward_spot_neighbors(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_spot, x_neighbors = x
        x_spot = self._apply_phi(x_spot, self.phi_spot)
        x_neighbors = self._apply_phi(x_neighbors, self.phi_spot, operation=Operation.MAX)
        x_phi = torch.cat((x_spot, x_neighbors), dim=1)
        return self._apply_rho(x_phi)

    def _forward_spot_subspot_neighbors(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_spot, x_subspot, x_neighbors = x
        x_subspot = self._apply_phi(x_subspot, self.phi_spot, operation=Operation.SUM)
        x_spot = self._apply_phi(x_spot, self.phi_spot)
        x_neighbors = self._apply_phi(x_neighbors, self.phi_spot, operation=Operation.MAX)
        x_phi = torch.cat((x_spot, x_subspot, x_neighbors), dim=1)
        return self._apply_rho(x_phi)

    def _apply_phi(self, x: torch.Tensor, phi_modules: nn.ModuleList, operation: Operation=Operation.NONE) -> torch.Tensor:
        batch_size = x.shape[0]
        sample_size = x.shape[1]
        x = x.view(-1, self.hparams.input_size)
        x_phi = torch.stack([phi(x) for phi in phi_modules], dim=1)
        x_phi, _ = torch.median(x_phi, dim=1)
        x_phi = x_phi.view(batch_size, sample_size, -1)
        if operation == Operation.SUM:
            x_phi = x_phi.sum(dim=1)
        elif operation == Operation.MEAN:
            x_phi = x_phi.mean(dim=1)
        elif operation == Operation.MAX:
            x_phi, _ = x_phi.max(dim=1)
        elif operation == Operation.NONE:
            x_phi = x_phi.view(batch_size, -1)
        return x_phi

    def _apply_rho(self, x_phi: torch.Tensor) -> torch.Tensor:
        x_rho = torch.stack([rho(x_phi) for rho in self.rho], dim=1)
        x_rho = torch.mean(x_rho, dim=1)
        return x_rho

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return {'optimizer': optimizer}

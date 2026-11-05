"""Official sklearn StandardScaler target recipe, fitted on training rows only.

No target access in inverse_prediction. Tensor-only checkpoint payload avoids
pickled estimator objects; reconstruction never fits or updates statistics.
"""
from __future__ import annotations
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import sklearn.utils.extmath
import torch

SCALER_NAME='sklearn_StandardScaler_training_only_resampling0_v1'

def array_sha(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(json.dumps(list(a.shape)).encode());h.update(a.tobytes())
    return h.hexdigest()

def source_paths():
    return [Path(__file__).resolve(),Path(inspect.getsourcefile(StandardScaler)).resolve(),
            Path(inspect.getsourcefile(sklearn.utils.extmath)).resolve()]

def validate_state(state, genes=None):
    expected={'name','sklearn_version','mean','var','scale','n_samples_seen','n_features','training_indices_sha256','training_targets_sha256'}
    if set(state)!=expected or state['name']!=SCALER_NAME or state['sklearn_version']!=sklearn.__version__:
        raise ValueError('Target scaler recipe/environment changed')
    g=state['n_features']
    if not isinstance(g,int) or g<1 or (genes is not None and genes!=g) or state['n_samples_seen']<1:
        raise ValueError('Invalid fitted scaler dimensions/count')
    for key in ('mean','var','scale'):
        x=state[key]
        if not isinstance(x,torch.Tensor) or x.dtype!=torch.float64 or x.shape!=(g,) or not torch.isfinite(x).all():
            raise ValueError('Invalid scaler tensor: '+key)
    if not (state['var']>=0).all() or not (state['scale']>0).all():raise ValueError('Invalid scaler variance/scale')
    for key in ('training_indices_sha256','training_targets_sha256'):
        if len(state[key])!=64:raise ValueError('Missing training-only scaler binding')
    return state

def state_sha(state):
    validate_state(state)
    plain={k:v.tolist() if isinstance(v,torch.Tensor) else v for k,v in state.items()}
    return hashlib.sha256(json.dumps(plain,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def fit_training(y, train):
    labels=np.ascontiguousarray(y[train])
    if labels.dtype!=np.float32 or labels.ndim!=2 or not np.isfinite(labels).all():
        raise ValueError('Scaler fit requires finite float32 training labels only')
    # This exact class is used by pinned upstream DeepSpot, including the
    # sklearn numerical constant-feature test and unit-scale replacement.
    scaler=StandardScaler().fit(labels)
    state={'name':SCALER_NAME,'sklearn_version':sklearn.__version__,
           'mean':torch.from_numpy(scaler.mean_.copy()),'var':torch.from_numpy(scaler.var_.copy()),
           'scale':torch.from_numpy(scaler.scale_.copy()),'n_samples_seen':int(scaler.n_samples_seen_),
           'n_features':int(scaler.n_features_in_),'training_indices_sha256':array_sha(train),
           'training_targets_sha256':array_sha(labels)}
    return validate_state(state,labels.shape[1])

def verify_training_binding(state,y,train):
    validate_state(state,y.shape[1])
    if (state['n_samples_seen']!=len(train) or state['training_indices_sha256']!=array_sha(train)
        or state['training_targets_sha256']!=array_sha(y[train])):
        raise ValueError('Fitted scaler training-only bindings differ')

def estimator(state):
    validate_state(state)
    scaler=StandardScaler()
    scaler.mean_=state['mean'].numpy().copy();scaler.var_=state['var'].numpy().copy()
    scaler.scale_=state['scale'].numpy().copy();scaler.n_features_in_=state['n_features']
    scaler.n_samples_seen_=np.int64(state['n_samples_seen'])
    return scaler

def transform_training(labels,state):
    result=estimator(state).transform(np.ascontiguousarray(labels,dtype=np.float32),copy=True)
    return torch.from_numpy(result)

def inverse_prediction(predictions,state):
    """Fixed training-artifact transform; accepts no molecular measurement."""
    if predictions.device.type!='cpu' or predictions.dtype!=torch.float32 or predictions.requires_grad:
        raise ValueError('Inverse prediction requires detached CPU float32 output')
    result=estimator(state).inverse_transform(predictions.numpy(),copy=True)
    return torch.from_numpy(result)

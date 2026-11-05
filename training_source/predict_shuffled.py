"""Frozen DeepSpot on a new image universe; no target/split/library-size input API."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
import argparse,copy,fcntl,importlib,json,signal,sys,math,shutil
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parents[2];V=P/'V2_BIOLOGICAL_REDESIGN';sys.path.insert(0,str(P/'03_code_environment/src'))
import train_head as h
import deepspot_target_scaler_v1 as scaling
STOP=False
def stop(*args):
 global STOP;STOP=True

def main():
 ap=argparse.ArgumentParser()
 for n in ['features','rows','context','training-run','run']:ap.add_argument('--'+n,required=True)
 ap.add_argument('--qualification-batches',type=int);ap.add_argument('--control',choices=['shuffled'],required=True);ap.add_argument('--pause-after',type=int);ap.add_argument('--resume');a=ap.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 source=importlib.import_module('train_shuffled')
 run=Path(a.run);run.mkdir(parents=True,exist_ok=True);lock=open(run/'.run.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if not a.resume:assert not [p for p in run.iterdir() if p.name!='.run.lock' and not p.name.startswith('._')]
 else:assert not (run/'PAUSE_REQUEST').exists()
 assert shutil.disk_usage(run).free>110*1024**3
 training=Path(a.training_run);assert (training/'results/manifest.json').exists();ptr=json.loads((training/'CURRENT.json').read_text());ref=ptr['current'];weightpath=training/ref['path'];saved=h.checked_load(weightpath,ref['sha256']);assert saved['phase']=='complete';cfg=saved['bindings']['config']
 assert (cfg.get('context_control')=='all_real_neighbor_slots_replaced_by_query_slot0')==(a.control=='self')
 assert saved['best_model'] is not None and all(torch.equal(saved['model'][k],saved['best_model'][k]) for k in saved['model'])
 assert saved['bindings']['environment']==h.version_binding()
 for p,d in saved['bindings']['source_code_sha256'].items():assert h.digest(p)==d
 scaler=copy.deepcopy(saved['target_scaler']);scaling.validate_state(scaler)
 with np.load(training/'results/predictions.npz') as z:genes=z['gene_names'].copy();gene_ids=z['gene_ids'].copy()
 paths={'features':Path(a.features),'rows':Path(a.rows),'context':Path(a.context)};bound={k:h.digest(p) for k,p in paths.items()}
 with np.load(paths['rows']) as z:
  allowed={'row_ids','samples','donors','coords','grid_coords'};assert set(z.files)==allowed;rows={k:z[k].copy() for k in z.files}
 X=np.load(paths['features'],mmap_mode='r');n=len(rows['row_ids']);assert X.shape==(n,10,768) and len(np.unique(rows['row_ids']))==n
 with np.load(paths['context']) as z:assert set(z.files)=={'row_ids','indptr','indices'};assert np.array_equal(z['row_ids'],rows['row_ids']);offset=z['indptr'];neighbors=z['indices']
 assert offset.shape==(n+1,) and offset[0]==0 and offset[-1]==len(neighbors) and np.all(np.diff(offset)>=0) and np.max(np.diff(offset))==4
 lookup={(str(s),int(r),int(c)):i for i,(s,(r,c)) in enumerate(zip(rows['samples'],rows['grid_coords']))}
 for i,(s,(r,c)) in enumerate(zip(rows['samples'],rows['grid_coords'])):
  expected=sorted(lookup[k] for dr in (-1,0,1) for dc in (-1,0,1) if (dr or dc) and (k:=(str(s),int(r)+dr,int(c)+dc)) in lookup);assert np.array_equal(neighbors[offset[i]:offset[i+1]],expected)
 source.set_shuffle(rows)
 image_universe_n=n
 if a.qualification_batches is not None:
  assert 1<=a.qualification_batches<=2; n=min(n,1024*a.qualification_batches)
 bindings={'images':{k:{'path':str(p.resolve()),'sha256':bound[k]} for k,p in paths.items()},'checkpoint':{'path':str(weightpath),'sha256':ref['sha256']},'source_code':{str(Path(__file__)):h.digest(__file__),**saved['bindings']['source_code_sha256']},'environment':h.version_binding(),'config':{'batch':1024,'control':a.control,'padding':4,'target_free_new_image_inputs':True,'image_universe_n':image_universe_n,'prediction_rows':n,'qualification_only':a.qualification_batches is not None}}
 model=source.DeepSpot(input_size=768,output_size=len(genes),loss_func='mse',random_seed=cfg['seed'],scaler=None,spot_context='spot_subspot_neighbors').cpu();model.load_state_dict(saved['best_model']);model.eval();del saved
 cp=h.Checkpoints(run)
 if a.resume:state=cp.resume(a.resume,bindings);h.restore_rng(state['rng'])
 else:
  state={'schema_version':1,'bindings':bindings,'phase':'predict','epoch':0,'update':0,'batch_cursor':0,'prediction_cursor':0,'budget_total_updates':math.ceil(n/1024),'predictions':torch.zeros((n,len(genes)),dtype=torch.float64),'scaler':scaler,'rng':h.rng_state(),'immutable_model_reference':ref};h.atomic_json(run/'CONFIG.json',bindings,immutable=True);cp.save(state)
 assert scaling.state_sha(state['scaler'])==scaling.state_sha(scaler)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 def pause():
  state['rng']=h.rng_state();ref=cp.save(state);h.update_status(run,'PAUSED',state,checkpoint=ref);return 75
 while state['prediction_cursor']<n:
  i=state['prediction_cursor'];end=min(n,i+1024);pred=scaling.inverse_prediction(source.prediction_batch(model,X,np.arange(i,end),offset,neighbors,4),scaler);assert pred.shape==(end-i,len(genes)) and bool(torch.isfinite(pred).all());state['predictions'][i:end]=pred.to(torch.float64);state['prediction_cursor']=end;state['update']+=1;h.update_status(run,'PREDICTING',state)
  if STOP or (run/'PAUSE_REQUEST').exists() or (a.pause_after is not None and state['update']>=a.pause_after):return pause()
 state['phase']='complete';state['rng']=h.rng_state();ref=cp.save(state)
 p=run/'predictions.npz'
 with p.open('xb') as f:np.savez_compressed(f,predictions=state['predictions'].numpy(),row_ids=rows['row_ids'][:n],gene_names=genes,gene_ids=gene_ids);f.flush();os.fsync(f.fileno())
 assert all(h.digest(p)==bound[k] for k,p in paths.items());h.atomic_json(run/'COMPLETE.json',{'status':'COMPLETE','predictions_sha256':h.digest(run/'predictions.npz'),'checkpoint':ref,'model_refit':False,'molecular_target_files_opened':False,'query_rows':n},immutable=True);h.update_status(run,'COMPLETE',state,checkpoint=ref);return 0
if __name__=='__main__':sys.exit(main())

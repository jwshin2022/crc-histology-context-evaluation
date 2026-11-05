from pathlib import Path
import csv, json
import numpy as np
BASE=Path(__file__).resolve().parent
OUT=BASE/'reproduced'
OUT.mkdir(exist_ok=True)
def write(name, records):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]))
        w.writeheader();w.writerows(records)
def graph(rows):
    lookup={(s,int(x),int(y)):i for i,(s,(x,y)) in enumerate(zip(rows['samples'],rows['grid_coords']))}
    assert len(lookup)==len(rows['samples'])
    edges=[]
    for (s,x,y),i in lookup.items():
        for dx,dy in [(2,0),(1,1),(1,-1)]:
            j=lookup.get((s,x+dx,y+dy))
            if j is not None:edges.append((i,j))
    return np.array(edges).T

with np.load(BASE/'evaluation_inputs.npz',allow_pickle=False) as z:
    y=z['measured'];eligible=z['eligible'];ids=z['row_ids']
    rows={k:z[k] for k in ['samples','donors','coords','grid_coords']}
    programs=list(z['programs']);variance=z['variance'];edge_den=z['edge_den']
    Y=z['training_scores'];mu=Y.mean(0)
    pred={k.removeprefix('prediction_'):z[k] for k in z.files if k.startswith('prediction_')}
assert np.allclose(Y.var(0),variance,atol=1e-12,rtol=0)
pred['training_mean']=np.broadcast_to(mu,y.shape)
a,b=graph(rows)
records=[]
for donor in np.unique(rows['donors']):
    donor_mask=eligible&(rows['donors']==donor)
    for group in ['donor',*np.unique(rows['samples'][donor_mask])]:
        mask=donor_mask.copy()
        if group!='donor':mask&=rows['samples']==group
        edges=mask[a]&mask[b]
        for model,p in pred.items():
            r=p-y;res=r[mask];bias=res.mean(0);mse=(res**2).mean(0);center=((res-bias)**2).mean(0)
            contrast=((r[a[edges]]-r[b[edges]])**2).mean(0)
            assert np.allclose(mse,bias**2+center,atol=1e-12)
            for j,program in enumerate(programs):
                records.append(dict(donor=donor,group=group,model=model,program=program,spots=int(mask.sum()),edges=int(edges.sum()),normalized_MSE=mse[j]/variance[j],normalized_contrast=contrast[j]/edge_den[j],bias=bias[j],normalized_squared_bias=bias[j]**2/variance[j],normalized_centered_MSE=center[j]/variance[j],train_mean=mu[j],test_mean=y[mask,j].mean(),test_variance=y[mask,j].var(),train_variance=variance[j]))
write('per_section_and_donor.csv',records)
table=[]
for donor in np.unique(rows['donors']):
    record={'donor':str(donor).split('_')[-1]}
    for model in ['training_mean','20261019_self','20261019_spatial']:
        vals=[x for x in records if x['donor']==donor and x['group']=='donor' and x['model']==model]
        for metric in ['normalized_MSE','normalized_contrast']:
            record[model+'_'+metric]=np.mean([x[metric] for x in vals])
    record['spatial_vs_mean_point_pct']=100*(record['20261019_spatial_normalized_MSE']/record['training_mean_normalized_MSE']-1)
    table.append(record)
write('training_mean_first_result.csv',table)
print(json.dumps(table,indent=2))
# Additional comparisons after the original confirmation. No model selection.
direction=np.abs(rows['grid_coords'][a,1]-rows['grid_coords'][b,1])
graph_records=[]
for donor in np.unique(rows['donors']):
    mask=eligible&(rows['donors']==donor);ee=mask[a]&mask[b]
    for kind,selected in [('four_input_directions',ee&(direction==1)),('two_omitted_directions',ee&(direction==0))]:
        for key,p in pred.items():
            r=p-y
            val=(((r[a[selected]]-r[b[selected]])**2).mean(0)/edge_den).mean()
            graph_records.append(dict(donor=donor,model=key,directions=kind,edges=int(selected.sum()),normalized_contrast=val))
write('graph_direction_errors.csv',graph_records)
pairs=[]
for donor in np.unique(rows['donors']):
    for seed in [20261019,20261104,20261105]:
        group=[x for x in records if x['donor']==donor and x['group']=='donor']
        get=lambda m,k:np.mean([x[k] for x in group if x['model']==m])
        pairs.append(dict(donor=donor,seed=seed,self_point=get(f'{seed}_self','normalized_MSE'),spatial_point=get(f'{seed}_spatial','normalized_MSE'),self_contrast=get(f'{seed}_self','normalized_contrast'),spatial_contrast=get(f'{seed}_spatial','normalized_contrast'),point_delta=get(f'{seed}_spatial','normalized_MSE')-get(f'{seed}_self','normalized_MSE'),bias_delta=get(f'{seed}_spatial','normalized_squared_bias')-get(f'{seed}_self','normalized_squared_bias'),center_delta=get(f'{seed}_spatial','normalized_centered_MSE')-get(f'{seed}_self','normalized_centered_MSE')))
write('nine_donor_seed_pairs.csv',pairs)
# Disjoint spatial blocks; sample blocks independently within each of the two fixed sections.
# Edge values are assigned by midpoint, so none are dropped at block borders.
B=2000
rng=np.random.default_rng(20261105)
bootstrap=[]
for width in [400,800,1200]:
 for shift in [0,width/2]:
  for donor in np.unique(rows['donors']):
   mask=eligible&(rows['donors']==donor)
   sections=np.unique(rows['samples'][mask])
   for seed in [20261019,20261104,20261105]:
    models=['training_mean',f'{seed}_self',f'{seed}_spatial']
    totals=np.zeros((B,3,2));pointcounts=np.zeros(B);edgecounts=np.zeros(B)
    blocknums=[]
    for section in sections:
     ix=np.flatnonzero(mask&(rows['samples']==section));ei=np.flatnonzero((rows['samples'][a]==section)&mask[a]&mask[b])
     pos=rows['coords'][ix];mids=(rows['coords'][a[ei]]+rows['coords'][b[ei]])/2
     origin=rows['coords'][rows['samples']==section].min(0)+shift
     pb=np.floor((pos-origin)/width).astype(int);eb=np.floor((mids-origin)/width).astype(int)
     keys=sorted(set(map(tuple,pb))|set(map(tuple,eb)));lookup={k:i for i,k in enumerate(keys)}
     pi=np.array([lookup[tuple(k)] for k in pb]);ej=np.array([lookup[tuple(k)] for k in eb]);nb=len(keys);blocknums.append(nb)
     pc=np.bincount(pi,minlength=nb);ec=np.bincount(ej,minlength=nb)
     multiplicities=rng.multinomial(nb,np.ones(nb)/nb,size=B)
     pointcounts+=multiplicities@pc;edgecounts+=multiplicities@ec
     for mi,model in enumerate(models):
      residual=pred[model]-y
      point=((residual[ix]**2)/variance).mean(1)
      contrast=(((residual[a[ei]]-residual[b[ei]])**2)/edge_den).mean(1)
      totals[:,mi,0]+=multiplicities@np.bincount(pi,weights=point,minlength=nb)
      totals[:,mi,1]+=multiplicities@np.bincount(ej,weights=contrast,minlength=nb)
    assert (pointcounts>0).all() and (edgecounts>0).all()
    totals[:,:,0]/=pointcounts[:,None];totals[:,:,1]/=edgecounts[:,None]
    for left,right in [(2,1),(2,0)]:
     for metric in [0,1]:
      delta=totals[:,left,metric]-totals[:,right,metric];lo,hi=np.quantile(delta,[.025,.975])
      bootstrap.append(dict(donor=donor,seed=seed,comparison='spatial_minus_'+('self' if right==1 else 'training_mean'),metric=['point','contrast'][metric],width_um=width,shift_um=shift,blocks_section1=blocknums[0],blocks_section2=blocknums[1],lower=lo,upper=hi,replicates=B,scope='within_observed_sections_conditional'))
write('spatial_block_bootstrap.csv',bootstrap)
print('Section decomposition, graph diagnostics and conditional block intervals completed.')

section_changes=[]
for donor in sorted(set(rows['donors'])):
 for section in sorted(set(rows['samples'][rows['donors']==donor])):
  for seed in [20261019,20261104,20261105]:
   part=[r for r in records if r['donor']==donor and r['group']==section]
   def avg(model,key):
    return np.mean([r[key] for r in part if r['model']==model])
   section_changes.append(dict(donor=donor,section=section,seed=seed,bias_delta=avg(f'{seed}_spatial','normalized_squared_bias')-avg(f'{seed}_self','normalized_squared_bias'),center_delta=avg(f'{seed}_spatial','normalized_centered_MSE')-avg(f'{seed}_self','normalized_centered_MSE')))
write('section_changes.csv',section_changes)

# Author: Don van Beusichem
"""Update derivative references, remove anchor geometries, preserve trained weights."""
import numpy as np,json
from pathlib import Path
from train import data,summarize
from neural import CoefficientNet,quantities
from response_study import Wmat,counterexample
P=Path(__file__).parent
# Save final force observables in original curve files and rescore JSON separately.
for n in [2,4,6]:
 a,Hp=data(n);r,H=[a[k] for k in ['r','H']]
 for file in P.glob(f'H{n}_n*_w*_p*_s*.npz'):
  z=dict(np.load(file));width=int(file.stem.split('_w')[1].split('_')[0]);q=quantities(CoefficientNet(H.shape[1],width),z['parameters'],r,H,Hp);z.update(q);np.savez_compressed(file,**z)
# Convergence report on unseen queries only.
old=json.loads((P/'convergence_results.json').read_text());rows=[]
for n in [4,6]:
 a,Hp=data(n);r,H,E,c,gap=[a[k] for k in ['r','H','E','c','gap']]
 for seed in [0,1,2]:
  keys=['anchors','width','penalty','seed','success','iterations','seconds','maxiter','gradient_inf'];o=next(x for x in old if x['system']==f'H{n}' and x['seed']==seed);mod={k:o[k] for k in keys};file=P/f'long_H{n}_s{seed}.npz';z=dict(np.load(file));rr,q=summarize(n,r,H,Hp,E,c,gap,mod,CoefficientNet(H.shape[1],24),z['p']);rows+=rr;z.update(q);np.savez_compressed(file,**z)
(P/'convergence_results.json').write_text(json.dumps(rows,indent=2))
# Orbital-twist chart derivative and force covariance with final H'.
a,Hp=data(4);r,H,E,c,gap=[a[k] for k in ['r','H','E','c','gap']];step=1e-5;W=np.array([Wmat(4,x) for x in r]);Wp=np.array([(Wmat(4,x+step)-Wmat(4,x-step))/(2*step) for x in r]);Hg=np.einsum('bji,bjk,bkl->bil',W,H,W);Hpg=np.einsum('bji,bjk,bkl->bil',Wp,H,W)+np.einsum('bji,bjk,bkl->bil',W,Hp,W)+np.einsum('bji,bjk,bkl->bil',W,H,Wp);cg=np.einsum('bji,bj->bi',W,c)
g=json.loads((P/'gauge_results.json').read_text());rows=[]
for n in [9,17]:
 for seed in [0,1,2]:
  o=next(x for x in g['rows'] if x['anchors']==n and x['seed']==seed);keys=['anchors','width','penalty','seed','success','iterations'];mod={k:o[k] for k in keys};file=P/f'gauge_n{n}_s{seed}.npz';z=dict(np.load(file));rr,q=summarize(4,r,Hg,Hpg,E,cg,gap,mod,CoefficientNet(36,24),z['p']);rows+=rr;z.update(q);np.savez_compressed(file,**z)
g['rows']=rows;(P/'gauge_results.json').write_text(json.dumps(g,indent=2));(P/'counterexample.json').write_text(json.dumps(counterexample(),indent=2))

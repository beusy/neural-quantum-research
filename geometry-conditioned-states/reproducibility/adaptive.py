# Author: Don van Beusichem
"""Matched-stage Hamiltonian-access collocation, not ab-initio cost reduction."""
import numpy as np,json
from pathlib import Path
from train import core,data,summarize
from neural import CoefficientNet,fit,quantities
P=Path(__file__).parent
# A fixed independent midpoint query grid, shared by every acquisition rule.
import chemistry as ch
from scipy.linalg import eigh
testfile=P/'adaptive_test.npz'
if not testfile.exists():
 tr=np.linspace(1.0125,4.9875,160);th=[];td=[];te=[];tc=[];step=.001
 for z in tr:
  hh=ch.getH(4,z);ev,cv=eigh(hh);th.append(hh);te.append(ev[0]);tc.append(cv[:,0]);td.append((-ch.getH(4,z+2*step)+8*ch.getH(4,z+step)-8*ch.getH(4,z-step)+ch.getH(4,z-2*step))/(12*step))
 np.savez_compressed(testfile,r=tr,H=th,Hp=td,E=te,c=tc)
test=np.load(testfile);tr,tH,tHp,tE,tc=[test[k] for k in ['r','H','Hp','E','c']]
a,Hp=data(4);r,H,E,c,gap=[a[k] for k in ['r','H','E','c','gap']];base=core(4,3.);rows=[]
for seed in [0,1,2]:
 for method in ['uniform','random','residual','response']:
  rng=np.random.default_rng(seed);nodes=np.array([np.argmin(abs(r-x)) for x in [1,2,3,4,5]]).tolist();net=CoefficientNet(36,24);p=None
  for stage in range(3):
   opt,obj=fit(net,r[nodes],H[nodes],seed,base,maxiter=600,initial=p);p=opt.x
   if stage==2:break
   q=quantities(net,p,r,H,Hp)
   if method=='uniform':new=[1.5,3.5] if stage==0 else [2.5,4.5];nodes.extend([int(np.argmin(abs(r-x))) for x in new]);continue
   score=q['residual'].copy() if method=='residual' else abs(q['response']).copy()
   for _ in range(2):
    allowed=(r>=1)&(r<=5)&(np.min(abs(r[:,None]-r[nodes][None,:]),axis=1)>.15)
    idx=int(rng.choice(np.where(allowed)[0])) if method=='random' else int(np.argmax(np.where(allowed,score,-np.inf)));nodes.append(idx)
  mask=np.ones(len(tr),bool);q=quantities(net,p,tr,tH,tHp);F0=-np.einsum('bi,bij,bj->b',tc,tHp,tc)
  np.savez_compressed(P/f'adaptive_{method}_s{seed}.npz',p=p,selected_indices=nodes,**q)
  rows.append(dict(seed=seed,method=method,anchors=9,iterations_budget=1800,selected_distances=r[nodes].tolist(),success=bool(opt.success),E_MAE_mHa=float(np.mean(abs(q['E'][mask]-tE[mask]))*1000),F_RMSE_mHa_bohr=float(np.sqrt(np.mean((q['F'][mask]-F0[mask])**2))*1000)))
  print(rows[-1],flush=True)
(P/'adaptive_results.json').write_text(json.dumps(rows,indent=2))

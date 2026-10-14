# Author: Don van Beusichem
import experiments as ex
import numpy as np, scipy.linalg as la,json
PTH=ex.ROOT
out={}
# Force observability: Cartesian force labels include rigid-motion null directions.
x=np.load(PTH/'data/rmd17_ethanol.npz');R=x['coords'];Z=x['nuclear_charges'];P,D=ex.feature_derivatives(R[:5],Z,True)
out['force_jacobian_ranks']=[int(np.linalg.matrix_rank(d.reshape(27,-1),tol=1e-7)) for d in D]
# Strict chronology split based on original trajectory frame index, not archive order.
old=x['old_indices'];order=np.argsort(old);rr=np.random.default_rng(43)
tr=rr.choice(order[:70000],200,replace=False);va=rr.choice(order[70000:80000],100,replace=False);te=rr.choice(order[80000:],300,replace=False);ids=np.r_[tr,va,te]
np.savez(PTH/'chronological_splits.npz',train=tr,validation=va,test=te)
P,D=ex.feature_derivatives(R[ids],Z,True);E=x['energies'][ids];F=x['forces'][ids];cand=[]
for reg in [1e-6,1e-3,1.]:
 mod=ex.fit_scaled(P[:200],D[:200],E[:200],F[:200],1,reg);ep,fp=ex.apply(P[200:300],D[200:300],mod);cand.append((np.mean((ep-E[200:300])**2)+np.mean((fp-F[200:300])**2),reg,mod))
_,reg,mod=min(cand,key=lambda a:a[0]);ep,fp=ex.apply(P[300:],D[300:],mod);out['chronological_ethanol']=dict(reg=reg,**ex.metric(E[300:],F[300:],ep,fp))
# Force-only supervision cannot fix offsets in disconnected parts of support.
# Analytic C-infinity bump step: exactly zero gradient on observed domains.
# E_2=E_1+C*S(r), S=0 for r<=1.0, S=1 for r>=1.5.
# Here report a finite-C identity, no numerical fit is necessary.
out['disconnected_force_gauge']={'C_eV':2.,'left_force_difference':0.,'right_force_difference':0.,'relative_energy_offset_eV':2.}
# Analytic MLP mixed derivatives checked against parameter finite differences.
p=np.random.default_rng(9).normal(0,.2,73);r=np.array([.6,1.,1.4]);e,g,je,jg=ex.nn(r,p);errs=[]
for k in range(73):
 plus=p.copy();minus=p.copy();plus[k]+=1e-6;minus[k]-=1e-6
 ep,gp,*_=ex.nn(r,plus);em,gm,*_=ex.nn(r,minus);errs.append(max(np.max(abs((ep-em)/2e-6-je[:,k])),np.max(abs((gp-gm)/2e-6-jg[:,k]))))
out['MLP_mixed_derivative_check']=float(max(errs))
# Active selection: joint energy-derivative GP posterior uncertainty.
grid=np.linspace(.35,2.,180);test=np.linspace(.35,2.,500);et,gt=ex.morse(test);rows=[]
for seed in [0,1,2]:
 for acquisition in ['random','energy_variance','force_variance']:
  rr=np.random.default_rng(seed);idx=list(rr.choice(len(grid),4,replace=False))
  for budget in [4,8,16]:
   while len(idx)<budget:
    if acquisition=='random':k=int(rr.choice(np.setdiff1d(np.arange(len(grid)),idx)))
    else:
     xx=grid[idx];K,ky,kx,kxy=ex.kernel(xx,xx);K=np.block([[K,ky],[kx,kxy]])+1e-8*np.eye(2*len(xx));k,ky,kx,kxy=ex.kernel(grid,xx);B=np.c_[k,ky] if acquisition=='energy_variance' else np.c_[kx,kxy];prior=1 if acquisition=='energy_variance' else 1/.35**2
     score=prior-np.einsum('ij,ji->i',B,la.solve(K,B.T,assume_a='pos'));score[idx]=-np.inf;k=int(np.argmax(score))
    idx.append(k)
   pred=ex.gpfit(grid[idx],'EF');ep,gp=pred(test);rows.append(dict(seed=seed,acquisition=acquisition,n=budget,**ex.metric(et,gt,ep,gp)))
out['active_learning']=rows
(PTH/'diagnostics.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='active_learning'},indent=2))

# Author: Don van Beusichem
from core import *
import time
P.mkdir(exist_ok=True)
rows=[];validation=[]
for name,n,param in [('ising_ferro',8,.2),('ising_critical',8,1.),('ising_para',8,2.),('frustrated',8,.7),('ising_critical_N10',10,1.)]:
 H,states,spins=j1j2(n,param) if name=='frustrated' else tfim(n,param);e,target=ground(H,name!='frustrated',n);np.savez_compressed(P/f'{name}_reference.npz',states=states,spins=spins,H=H.toarray(),E=e,target=target)
 if name!='frustrated':
  k=(2*np.arange(n)+1)*np.pi/n;analytic=-np.sum(np.sqrt(1+param*param-2*param*np.cos(k)));validation.append(dict(system=name,ground_energy=float(e[0]),analytic_energy=float(analytic),absolute_error=float(abs(e[0]-analytic)),eigen_residual=float(np.linalg.norm(H@target-e[0]*target))))
 families=['RBM','MLP'] if name=='frustrated' else ['RBM'];widths=[4,8,16] if n==8 else [8]
 for family in families:
  for width in widths:
   for seed in [0,1,2]:
    for objective in ['energy','infidelity']:
     model=RBM(n,width) if family=='RBM' else SignedMLP(n,width);t=time.time();met,p,history,checkpoints=train(model,H,spins,target,e[0],seed,objective,maxiter=1200);met.update(system=name,family=family,seconds=time.time()-t);rows.append(met)
     key=f'{name}_{family}_w{width}_s{seed}_{objective}';np.savez_compressed(P/f'{key}.npz',p=p,**{f'checkpoint_{i}':v for i,v in checkpoints.items()});(P/f'{key}_history.json').write_text(json.dumps(history,indent=2));print({k:v for k,v in met.items() if k in ['system','family','width','seed','objective','E_error','infidelity','QGT_rank','success','iterations']},flush=True)
 (P/'training_results.json').write_text(json.dumps(rows,indent=2));(P/'validation.json').write_text(json.dumps(validation,indent=2))
# Analytic restriction for all positive coefficient ansätze in the Marshall chart.
a=np.load(P/'frustrated_reference.npz');v=a['target'];plus=float(np.sum(v[v>0]**2));minus=float(np.sum(v[v<0]**2));(P/'positive_bound.json').write_text(json.dumps(dict(positive_mass=plus,negative_mass=minus,best_possible_infidelity=min(plus,minus),formal_positive_logtable_QGT_rank=len(v)-1,dimension=len(v)),indent=2))

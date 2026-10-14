# Author: Don van Beusichem
from core import *
from scipy.special import rel_entr

def chain(logp,n,count,seed,global_flip=0.,initial=None,burn=1000):
 rng=np.random.default_rng(seed);state=(1<<n)-1 if initial is None else initial;draws=np.empty(count,int);accept=0;proposals=0
 for t in range(count+burn):
  for _ in range(n):
   proposed=state^((1<<n)-1) if rng.random()<global_flip else state^(1<<int(rng.integers(n)))
   if np.log(rng.random())<logp[proposed]-logp[state]:state=proposed;accept+=1
   proposals+=1
  if t>=burn:draws[t-burn]=state
 return draws,accept/proposals

def jsd(p,q):
 m=(p+q)/2;return float(.5*np.sum(rel_entr(p,m))+.5*np.sum(rel_entr(q,m)))

def frozen_sampling():
 H,states,spins=tfim(10,.2);e,c=ground(H,True,10);prob=c*c;logp=np.log(np.maximum(prob,1e-300));local=(H@c)/c;mag=spins.mean(1);rows=[]
 np.savez_compressed(P/'frozen_sampling_reference.npz',H=H.toarray(),target=c,E=e,probabilities=prob,magnetization=mag,local_energy=local)
 for method in ['iid','local_up','local_down','global_flip']:
  for seed in range(20):
   rng=np.random.default_rng(seed)
   if method=='iid':idx=rng.choice(len(c),4096,p=prob);accept=None
   else:idx,accept=chain(logp,10,4096,seed,.05 if method=='global_flip' else 0.,initial=0 if method=='local_down' else None)
   empirical=np.bincount(idx,minlength=len(c))/len(idx);ms=mag[idx];E=local[idx];flips=int(np.sum(np.sign(ms[1:])!=np.sign(ms[:-1])));mcse=float(np.std(ms,ddof=1)/np.sqrt(len(ms)))
   rows.append(dict(method=method,seed=seed,samples=len(idx),mean_magnetization=float(ms.mean()),magnetization_abs_error=float(abs(ms.mean()-prob@mag)),naive_magnetization_SE=mcse,mean_energy_error=float(E.mean()-e[0]),local_energy_variance=float(np.var(E)),total_variation=float(.5*np.sum(abs(empirical-prob))),JSD=jsd(empirical,prob),magnetization_sign_changes=flips,acceptance=accept))
   if seed==0:np.savez_compressed(P/f'samples_{method}.npz',indices=idx,magnetization=ms,local_energies=E)
  print('sampling',method,rows[-1],flush=True)
 (P/'sampling_results.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':frozen_sampling()

# Author: Don van Beusichem
from core import *
import scipy.linalg as la

def run():
 rows=[]
 for name in ['ising_ferro','ising_critical']:
  a=np.load(P/f'{name}_reference.npz');H=coo_matrix(a['H']).tocsr();sp=a['spins'];target=a['target'];E0=a['E'][0]
  for seed in [0,1,2]:
   for method in ['Adam','SR_isotropic','SR_diagonal','SR_block']:
    net=RBM(8,8);p=net.initial(seed);m=np.zeros_like(p);v=m.copy();history=[];evals=0;status='iteration_cap'
    for step in range(1,501):
     c,J,O=net.values(p,sp);hc=H@c;E=float(c@hc);g=2*J.T@(hc-E*c);evals+=1
     if method=='Adam':
      m=.9*m+.1*g;v=.999*v+.001*g*g;p-=.02*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
     else:
      S=J.T@J;scale=np.trace(S)/len(S)
      if method=='SR_block':
       tmp=np.zeros_like(S)
       for lo,hi in [(0,8),(8,16),(16,len(S))]:tmp[lo:hi,lo:hi]=S[lo:hi,lo:hi]
       S=tmp
      regularizer=1e-2*np.diag(np.diag(S))+1e-10*scale*np.eye(len(S)) if method=='SR_diagonal' else 1e-3*scale*np.eye(len(S))
      d=la.solve(S+regularizer,-g,assume_a='pos');eta=min(.1,.05/max(np.linalg.norm(J@d),1e-30),1/max(np.linalg.norm(d),1e-30));accepted=False
      for backtrack in range(25):
       pnew=p+eta*d;newc=net.values(pnew,sp,False);newE=float(newc@(H@newc));evals+=1
       if newE<=E+1e-4*eta*(g@d)+1e-14:p=pnew;accepted=True;break
       eta*=.5
      if not accepted:status='line_search_failed';break
     if step in [1,5,10,25,50,100,200,300,400,500]:
      cc=net.values(p,sp,False);history.append(dict(iteration=step,E_error=float(cc@(H@cc)-E0),infidelity=float(1-(cc@target)**2),magnetization=float((cc*cc)@sp.mean(1)),Hamiltonian_evaluations=evals))
    met,zz=measurements(net,p,sp,H,target,E0);met.update(system=name,seed=seed,method=method,iterations=step,Hamiltonian_evaluations=evals,status=status,mean_magnetization=float((zz['c']**2)@sp.mean(1)));rows.append(met);key=f'{name}_{method}_s{seed}';np.savez_compressed(P/f'optimizer_{key}.npz',p=p);(P/f'optimizer_{key}_history.json').write_text(json.dumps(history,indent=2));print({k:met[k] for k in ['system','method','seed','E_error','infidelity','Hamiltonian_evaluations']},flush=True)
 (P/'optimizer_results.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':run()

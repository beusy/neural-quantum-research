# Author: Don van Beusichem
from core import *
import scipy.linalg as la

def solve_preconditioned(S,g,lam,diagonal=False):
 sd=np.sqrt(np.maximum(np.diag(S),1e-300));C=S/sd[:,None]/sd[None,:]
 shift=np.ones(len(S))*lam if diagonal else lam/(sd*sd)
 return la.solve(C+np.diag(shift),-g/sd,assume_a='pos')/sd

def run():
 a=np.load(P/'ising_critical_reference.npz');H=coo_matrix(a['H']).tocsr();spins=a['spins'];model=RBM(8,8);p=np.load(P/'ising_critical_RBM_w8_s0_energy.npz')['checkpoint_25'];met,z=measurements(model,p,spins,H,a['target'],a['E'][0]);J,S=z['J'],z['S'];g=2*J.T@z['residual'];rows=[];rng=np.random.default_rng(19);direction=rng.uniform(-1,1,len(p));baseline={}
 for extent in [0.,1.,3.,6.]:
  scales=10**(extent*direction);Sf=scales[:,None]*S*scales[None,:];gf=g*scales;w=la.eigvalsh(Sf);sd=np.sqrt(np.diag(Sf));corr=Sf/sd[:,None]/sd[None,:];wc=la.eigvalsh(corr)
  for method in ['isotropic_shift','diagonal_shift']:
   lam=1e-3 if method=='diagonal_shift' else 1e-3*np.trace(Sf)/len(Sf)
   df=solve_preconditioned(Sf,gf,lam,method=='diagonal_shift');d=scales*df;tang=J@d;delta=d*.02/np.linalg.norm(tang);c=model.values(p+delta,spins,False);newE=c@(H@c)
   if extent==0:baseline[method]=(tang.copy(),delta.copy())
   v=baseline[method][0];cos=float(tang@v/(np.linalg.norm(tang)*np.linalg.norm(v)));error=float(np.linalg.norm(delta-baseline[method][1])/np.linalg.norm(baseline[method][1]))
   rows.append(dict(log10_coordinate_extent=extent,method=method,QGT_threshold_rank=int(np.sum(w>max(w[-1]*1e-8,1e-12))),correlation_threshold_rank=int(np.sum(wc>max(wc[-1]*1e-8,1e-12))),physical_direction_cosine=cos,relative_physical_parameter_step_change=error,finite_energy_change=float(newE-met['E'])))
 (P/'coordinate_results.json').write_text(json.dumps(rows,indent=2));print(rows)
 # Uniform positive log-table state: full local rank despite a global sign restriction.
 D=70;prob=np.ones(D)/D;S=np.diag(prob)-np.outer(prob,prob);eig=la.eigvalsh(S);(P/'positive_logtable.json').write_text(json.dumps(dict(dimension=D,rank=int(np.sum(eig>1e-12)),nonzero_eigenvalue=1/D,ground_state_best_infidelity=.5),indent=2))
if __name__=='__main__':run()

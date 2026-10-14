# Author: Don van Beusichem
from core import *
import scipy.linalg as la

def stats(O,el,idx):
 X=O[idx];X=X-X.mean(0);ec=el[idx]-el[idx].mean();return X.T@X/len(idx),2*X.T@ec/len(idx)

def run():
 a=np.load(P/'ising_critical_reference.npz');H=coo_matrix(a['H']).tocsr();spins=a['spins'];target=a['target'];E0=a['E'][0];net=RBM(8,8);saved=np.load(P/'ising_critical_RBM_w8_s0_energy.npz');rows=[];meta=[];popdata={}
 for checkpoint in [5,25]:
  p=saved[f'checkpoint_{checkpoint}'];met,z=measurements(net,p,spins,H,target,E0);c,J,O,S=z['c'],z['J'],z['O'],z['S'];el=(H@c)/c;g=2*J.T@z['residual'];scale=np.trace(S)/len(S);ridge=1e-3*scale;dref=la.solve(S+ridge*np.eye(len(S)),-g,assume_a='pos');vref=J@dref;vref/=np.linalg.norm(vref);meta.append(dict(checkpoint=checkpoint,**met));popdata[f'S_{checkpoint}']=S;popdata[f'g_{checkpoint}']=g;popdata[f'p_{checkpoint}']=p
  for shots in [16,64,256,1024]:
   for rep in range(100):
    rng=np.random.default_rng(100000*checkpoint+1000*shots+rep);idx=rng.choice(len(c),shots,p=c*c);Sh,gh=stats(O,el,idx);directions={
     'population':dref,
     'population_small_ridge':la.solve(S+1e-6*scale*np.eye(len(S)),-g,assume_a='pos'),
     'population_large_ridge':la.solve(S+1e-2*scale*np.eye(len(S)),-g,assume_a='pos'),
     'population_diagonal_shift':la.solve(S+1e-2*np.diag(np.diag(S))+1e-10*scale*np.eye(len(S)),-g,assume_a='pos'),
     'joint_ridge':la.solve(Sh+ridge*np.eye(len(S)),-gh,assume_a='pos'),
     'metric_only_noise':la.solve(Sh+ridge*np.eye(len(S)),-g,assume_a='pos'),
     'gradient_only_noise':la.solve(S+ridge*np.eye(len(S)),-gh,assume_a='pos'),
     'joint_small_ridge':la.solve(Sh+1e-6*scale*np.eye(len(S)),-gh,assume_a='pos'),
     'joint_large_ridge':la.solve(Sh+1e-2*scale*np.eye(len(S)),-gh,assume_a='pos'),
     'joint_diagonal_shift':la.solve(Sh+1e-2*np.diag(np.diag(Sh))+1e-10*scale*np.eye(len(S)),-gh,assume_a='pos')}
    Sb=np.zeros_like(Sh)
    for lo,hi in [(0,8),(8,16),(16,len(S))]:Sb[lo:hi,lo:hi]=Sh[lo:hi,lo:hi]
    directions['joint_block']=la.solve(Sb+ridge*np.eye(len(S)),-gh,assume_a='pos')
    # Exploratory two-half directional consistency filter, not claimed as novel.
    Sa,ga=stats(O,el,idx[:shots//2]);Sv,gv=stats(O,el,idx[shots//2:]);lam,U=la.eigh(Sa);lv=np.einsum('ij,ij->j',U,Sv@U);fa=U.T@ga;fv=U.T@gv
    keep=(lam>scale*1e-6)&(lv>.25*lam)&(lv<4*lam)&(fa*fv>0);coef=np.where(keep,-.5*(fa+fv)/(.5*(lam+lv)+ridge),0.);directions['split_filter']=U@coef
    eig=np.linalg.eigvalsh(Sh);rank=int(np.sum(eig>max(eig[-1]*1e-8,1e-12)))
    for method,d in directions.items():
     tang=J@d;norm=np.linalg.norm(tang);cos=float(tang@vref/max(norm,1e-30))
     for radius in [.002,.01,.02]:
      delta=d*(radius/max(norm,1e-30));lin=float(g@delta);newc=net.values(p+delta,spins,False);change=float(newc@(H@newc)-met['E']);actual=np.sqrt(max(0,2*(1-abs(c@newc))))
      rows.append(dict(checkpoint=checkpoint,shots=shots,rep=rep,method=method,radius=radius,unique_configurations=int(len(np.unique(idx))),QGT_rank=rank,metric_relative_error=float(np.linalg.norm(Sh-S)/np.linalg.norm(S)),gradient_relative_error=float(np.linalg.norm(gh-g)/np.linalg.norm(g)),linear_energy_change=lin,linear_uphill=bool(lin>1e-12),finite_energy_change=change,finite_uphill=bool(change>1e-12),physical_direction_cosine=cos,parameter_step_norm=float(np.linalg.norm(delta)),actual_projective_chord=float(actual)))
   print('QGT',checkpoint,shots,flush=True)
 (P/'qgt_results.json').write_text(json.dumps(rows,indent=2));(P/'qgt_checkpoints.json').write_text(json.dumps(meta,indent=2));np.savez_compressed(P/'qgt_population.npz',**popdata)

if __name__=='__main__':run()

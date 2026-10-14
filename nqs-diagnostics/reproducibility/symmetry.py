# Author: Don van Beusichem
from core import *
class SymRBM(RBM):
 name='Z2_projected_RBM'
 def values(self,p,spins,jac=True):
  if not jac:
   a=super().values(p,spins,False);v=a+a[::-1];return v/np.linalg.norm(v)
  c,J,O=super().values(p,spins,True);v=c+c[::-1];Jv=J+J[::-1];norm=np.linalg.norm(v);c=v/norm;J=Jv/norm-c[:,None]*(c@Jv/norm);O=J/c[:,None];return c,J,O
if __name__=='__main__':
 a=np.load(P/'ising_ferro_reference.npz');H=coo_matrix(a['H']).tocsr();spins=a['spins'];target=a['target'];E0=a['E'][0];rows=[]
 for width in [4,8,16]:
  for seed in [0,1,2]:
   model=SymRBM(8,width);met,p,hist,cp=train(model,H,spins,target,E0,seed,'energy',1200);c=model.values(p,spins,False);met.update(mean_magnetization=float((c*c)@spins.mean(1)),parity_expectation=float(c@c[::-1]));rows.append(met);np.savez_compressed(P/f'symmetry_w{width}_s{seed}.npz',p=p);print(met,flush=True)
 (P/'symmetry_results.json').write_text(json.dumps(rows,indent=2))

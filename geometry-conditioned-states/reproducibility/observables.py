# Author: Don van Beusichem
import chemistry as ch,numpy as np,json
from pathlib import Path
P=Path(__file__).parent
rows=[]
for n in [2,4,6]:
 a=np.load(P/f'H{n}_data.npz');r=a['r'];ref=a['c'];f=np.load(P/f'H{n}_n9_w{12 if n==2 else 24}_p0_s0.npz');pred=f['c'];ops,_,D=ch.operators(n);o=ops.astype(int)
 def gamma(cs):
  g=np.zeros((len(cs),n,n));values=cs[:,o[:,0]]*cs[:,o[:,1]]*ops[:,-1]
  for i in range(n):
   for j in range(n):g[:,i,j]=values[:,(o[:,2]==i)&(o[:,3]==j)].sum(1)
  return g
 g0=gamma(ref);g=gamma(pred);norm=np.linalg.norm(g-g0,axis=(1,2));seen=np.any(abs(r[:,None]-np.linspace(1,5,9))<1e-8,axis=1);mask=(r>=1)&(r<=5)&~seen
 rows.append(dict(n=n,one_RDM_Frobenius_mean_ID=float(np.mean(norm[mask])),trace_error_max=float(np.max(abs(np.trace(g,axis1=1,axis2=2)-n))),hermiticity_error_max=float(np.max(abs(g-g.transpose(0,2,1))))))
 if n==4:np.savez_compressed(P/'occupations_H4.npz',r=r,exact=np.linalg.eigvalsh(g0),neural=np.linalg.eigvalsh(g))
(P/'observables.json').write_text(json.dumps(rows,indent=2));print(rows)

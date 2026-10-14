# Author: Don van Beusichem
import numpy as np,json,time
from scipy.linalg import eigh
from scipy.interpolate import CubicSpline
from pathlib import Path
import chemistry as chem
from neural import CoefficientNet,fit,quantities
P=Path(__file__).parent

def core(n,r):
 R=chem.geometry(n,r);S,h,g,en=chem.integrals(R);X=chem.lowdin(S);hh,gg=chem.transform(h,g,X);eps,U=eigh(hh);_,occ=chem.basis(n);a=np.array([np.linalg.det(U[np.ix_(i,range(n//2))]) for i in occ]);return np.outer(a,a).ravel()
def data(n):
 a=np.load(P/f'H{n}_data.npz');r=a['r'];H=a['H'];hpfile=P/f'H{n}_derivatives.npz'
 if hpfile.exists():Hp=np.load(hpfile)['Hp']
 else:
  h=1e-4;Hp=[]
  for i,x in enumerate(r):
   Hp.append((chem.getH(n,x+h)-chem.getH(n,x-h))/(2*h))
  Hp=np.array(Hp);np.savez_compressed(hpfile,Hp=Hp)
 return a,Hp

def summarize(n,r,H,Hp,E,c,gap,model,net,p):
 q=quantities(net,p,r,H,Hp);err=q['E']-E;F0=-np.einsum('bi,bij,bj->b',c,Hp,c);fid=np.sum(c*q['c'],axis=1)**2
 rows=[]
 trainr=np.linspace(1,5,model['anchors']);seen=np.any(abs(r[:,None]-trainr[None,:])<1e-8,axis=1)
 for name,mask in [('ID',(r>=1)&(r<=5)&~seen),('anchors',seen),('compressed',r<1),('stretched',r>5)]:
  if not mask.any():continue
  er=err[mask];fr=q['F'][mask]-F0[mask];fh=q['F_frozen'][mask]-F0[mask]
  rows.append(dict(model,system=f'H{n}',region=name,count=int(mask.sum()),E_MAE_mHa=float(np.mean(abs(er))*1000),E_max_mHa=float(np.max(abs(er))*1000),E_min_Ha=float(np.min(er)),F_RMSE_mHa_bohr=float(np.sqrt(np.mean(fr*fr))*1000),F_frozen_RMSE_mHa_bohr=float(np.sqrt(np.mean(fh*fh))*1000),response_RMS_mHa_bohr=float(np.sqrt(np.mean(q['response'][mask]**2))*1000),infidelity_mean=float(np.mean(1-fid[mask])),residual_RMS_Ha=float(np.sqrt(np.mean(q['residual'][mask]**2)))))
 return rows,q

def run(n):
 a,Hp=data(n);r,H,E,c,gap=[a[k] for k in ['r','H','E','c','gap']];out=[];curves={};base=core(n,3.)
 configs=[(5,24,0.),(9,24,0.),(17,24,0.),(9,8,0.),(9,48,0.),(9,24,1.),(9,24,10.)]
 if n==2:configs=[(5,12,0.),(9,12,0.),(9,12,1.)]
 if n==6:configs=[(9,24,0.),(17,24,0.),(9,24,1.)]
 for anchors,width,penalty in configs:
  trainr=np.linspace(1,5,anchors);idx=np.array([np.argmin(abs(r-x)) for x in trainr]);assert np.max(abs(r[idx]-trainr))<1e-9
  for seed in [0,1,2]:
   t=time.time();net=CoefficientNet(len(base),width);opt,obj=fit(net,r[idx],H[idx],seed,base,penalty=penalty,maxiter=1800)
   model=dict(anchors=anchors,width=width,penalty=penalty,seed=seed,success=bool(opt.success),iterations=opt.nit,evaluations=opt.nfev,objective=float(opt.fun),seconds=time.time()-t)
   rows,q=summarize(n,r,H,Hp,E,c,gap,model,net,opt.x);out.extend(rows)
   key=f'n{anchors}_w{width}_p{penalty:g}_s{seed}';np.savez_compressed(P/f'H{n}_{key}.npz',parameters=opt.x,train_indices=idx,**q)
   print(n,key,rows[0],flush=True)
 (P/f'H{n}_results.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':
 import sys
 for n in ([int(x) for x in sys.argv[1:]] or [2,4,6]):run(n)

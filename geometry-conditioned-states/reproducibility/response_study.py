# Author: Don van Beusichem
"""Convergence, orbital-chart covariance, and energy/force counterexample."""
import numpy as np,json,time
from scipy.linalg import eigh
from scipy.interpolate import CubicSpline
from pathlib import Path
import chemistry as ch
from neural import CoefficientNet,fit,quantities
from train import data,core,summarize
P=Path(__file__).parent

def wedge(n,Q):
 _,occ=ch.basis(n);A=np.array([[np.linalg.det(Q[np.ix_(i,j)]) for j in occ] for i in occ]);return np.kron(A,A)
def Wmat(n,r,omega=3*np.pi/4):
 theta=omega*(r-1);Q=np.eye(n);Q[:2,:2]=[[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]];return wedge(n,Q)
def gauges():
 a,Hp=data(4);r,H,E,c=[a[k] for k in ['r','H','E','c']];h=1e-5
 W=np.array([Wmat(4,x) for x in r]);Wp=np.array([(Wmat(4,x+h)-Wmat(4,x-h))/(2*h) for x in r]);Hg=np.einsum('bji,bjk,bkl->bil',W,H,W);Hpg=np.einsum('bji,bjk,bkl->bil',Wp,H,W)+np.einsum('bji,bjk,bkl->bil',W,Hp,W)+np.einsum('bji,bjk,bkl->bil',W,H,Wp)
 rows=[];base=Wmat(4,3.).T@core(4,3.)
 for n in [9,17]:
  idx=np.array([np.argmin(abs(r-x)) for x in np.linspace(1,5,n)])
  for seed in [0,1,2]:
   net=CoefficientNet(36,24);opt,obj=fit(net,r[idx],Hg[idx],seed,base,maxiter=1800);model=dict(anchors=n,width=24,penalty=0.,seed=seed,success=bool(opt.success),iterations=opt.nit)
   row,q=summarize(4,r,Hg,Hpg,E,np.einsum('bji,bj->bi',W,c),a['gap'],model,net,opt.x);rows+=row;np.savez_compressed(P/f'gauge_n{n}_s{seed}.npz',p=opt.x,**q)
   print('gauge',n,seed,row[0],flush=True)
 # Fixed approximate PHYSICAL curve represented in both charts: total derivative covariance.
 f=np.load(P/'H4_n9_w24_p0_s0.npz');v=f['c'];dv=f['dc'];vg=np.einsum('bji,bj->bi',W,v);dvg=np.einsum('bji,bj->bi',Wp,v)+np.einsum('bji,bj->bi',W,dv)
 E0=np.einsum('bi,bij,bj->b',v,H,v);res0=np.einsum('bij,bj->bi',H,v)-E0[:,None]*v;resg=np.einsum('bij,bj->bi',Hg,vg)-E0[:,None]*vg
 frozen=np.einsum('bi,bij,bj->b',v,Hp,v);fg=np.einsum('bi,bij,bj->b',vg,Hpg,vg);resp=2*np.sum(dv*res0,axis=1);rg=2*np.sum(dvg*resg,axis=1)
 check={'total_derivative_covariance_max':float(np.max(abs((frozen+resp)-(fg+rg)))),'frozen_derivative_chart_change_RMS':float(np.sqrt(np.mean((frozen-fg)**2))),'energy_covariance_max':float(np.max(abs(E0-np.einsum('bi,bij,bj->b',vg,Hg,vg)))),'orbital_transform_orthogonality_max':float(np.max(abs(np.einsum('bji,bjk->bik',W,W)-np.eye(36))))}
 return rows,check

def convergence():
 rows=[]
 for n in [4,6]:
  a,Hp=data(n);r,H,E,c=[a[k] for k in ['r','H','E','c']];base=core(n,3.);idx=np.array([np.argmin(abs(r-x)) for x in np.linspace(1,5,9)])
  for seed in [0,1,2]:
   net=CoefficientNet(len(base),24);t=time.time();opt,obj=fit(net,r[idx],H[idx],seed,base,maxiter=7200)
   model=dict(anchors=9,width=24,penalty=0.,seed=seed,success=bool(opt.success),iterations=opt.nit,seconds=time.time()-t,maxiter=7200,gradient_inf=float(np.max(abs(opt.jac))))
   row,q=summarize(n,r,H,Hp,E,c,a['gap'],model,net,opt.x);rows+=row;np.savez_compressed(P/f'long_H{n}_s{seed}.npz',p=opt.x,**q);print('long',n,seed,row[0],flush=True)
 return rows

def counterexample():
 # Exact eigenstates provide an analytic control; not fitted neural predictions.
 rs=np.linspace(1,5,2001);a=np.load(P/'H4_data.npz');H=CubicSpline(a['r'],a['H'])(rs);ev=eigh(H[0])[0] # ensure scipy available
 vals,vec=np.linalg.eigh(H);gap=vals[:,1]-vals[:,0];Hp=CubicSpline(a['r'],a['H'])(rs,1);gp=np.einsum('bi,bij,bj->b',vec[:,:,1],Hp,vec[:,:,1])-np.einsum('bi,bij,bj->b',vec[:,:,0],Hp,vec[:,:,0]);epsilon=.02;out=[]
 for omega in [1.,10.,100.]:
  alpha=epsilon*np.sin(omega*rs);de=np.sin(alpha)**2*gap;df=-(np.sin(2*alpha)*epsilon*omega*np.cos(omega*rs)*gap+np.sin(alpha)**2*gp)
  out.append(dict(epsilon=epsilon,omega=omega,E_max_mHa=float(np.max(de)*1000),F_RMSE_mHa_bohr=float(np.sqrt(np.mean(df*df))*1000)))
 return out
if __name__=='__main__':
 rows,check=gauges();(P/'gauge_results.json').write_text(json.dumps(dict(rows=rows,checks=check),indent=2));long=convergence();(P/'convergence_results.json').write_text(json.dumps(long,indent=2));(P/'counterexample.json').write_text(json.dumps(counterexample(),indent=2))

# Author: Don van Beusichem
import numpy as np,json,time
from scipy.optimize import minimize
from scipy.linalg import eigh
from pathlib import Path
import chemistry as ch
from train import core
P=Path(__file__).parent
class Net2D:
 def __init__(self,width=32,dim=36):self.k=width;self.d=dim
 def unpack(self,p):
  k,d=self.k,self.d;return p[:2*k].reshape(2,k),p[2*k:3*k],p[3*k:3*k+k*d].reshape(k,d),p[3*k+k*d:]
 def initial(self,seed,base):
  rr=np.random.default_rng(seed);return np.r_[rr.normal(0,.6,2*self.k),rr.normal(0,.4,self.k),rr.normal(0,.01,self.k*self.d),base]
 def forward(self,X,p):
  A,b,W,c=self.unpack(p);t=np.tanh(X@A+b);s=1-t*t;return t@W+c,np.einsum('bk,ak,kd->bad',s,A,W)/1.5,(X,t,s,W)
 def backward(self,g,cache):
  X,t,s,W=cache;dz=(g@W.T)*s;return np.r_[(X.T@dz).ravel(),dz.sum(0),(t.T@g).ravel(),g.sum(0)]
def build():
 f=P/'dimer_data.npz'
 if f.exists():return np.load(f)
 grid=np.r_[.8,np.linspace(1,4,13),4.4];coords=np.array([(a,b) for a in grid for b in grid]);Hs=[];Es=[];Cs=[];Hps=[];gaps=[];h=1e-4
 for i,x in enumerate(coords):
  H=ch.getH(4,x,'dimer');e,c=eigh(H);der=[]
  for j in range(2):
   xp=x.copy();xm=x.copy();xp[j]+=h;xm[j]-=h;der.append((ch.getH(4,xp,'dimer')-ch.getH(4,xm,'dimer'))/(2*h))
  Hs.append(H);Es.append(e[0]);Cs.append(c[:,0]);Hps.append(der);gaps.append(e[1]-e[0])
  if i%50==0:print('dimer data',i,flush=True)
 np.savez_compressed(f,x=coords,H=Hs,E=Es,c=Cs,Hp=Hps,gap=gaps);return np.load(f)
def run():
 a=build();x,H,E,c,Hp=[a[k] for k in ['x','H','E','c','Hp']];X=(x-2.5)/1.5;rows=[];base=core(4,2.5);F0=-np.einsum('bi,bkij,bj->bk',c,Hp,c)
 for anchors in [3,5,7]:
  vals=np.linspace(1,4,anchors);train=np.array([i for i,z in enumerate(x) if np.min(abs(vals-z[0]))<1e-8 and np.min(abs(vals-z[1]))<1e-8]);assert len(train)==anchors**2
  for seed in [0,1,2]:
   t0=time.time();net=Net2D();p0=net.initial(seed,base)
   def obj(p):
    v,_,cache=net.forward(X[train],p);n=(v*v).sum(1);Hv=np.einsum('bij,bj->bi',H[train],v);en=(v*Hv).sum(1)/n;g=2*(Hv-en[:,None]*v)/n[:,None]/len(train)
    return en.mean()+1e-10*(p*p).sum(),net.backward(g,cache)+2e-10*p
   opt=minimize(obj,p0,jac=True,method='L-BFGS-B',options={'maxiter':2500,'ftol':1e-13,'gtol':1e-8,'maxcor':30,'maxls':40})
   v,dv,_=net.forward(X,opt.x);n=(v*v).sum(1);Hv=np.einsum('bij,bj->bi',H,v);en=(v*Hv).sum(1)/n;res=Hv-en[:,None]*v;fr=-np.einsum('bi,bkij,bj->bk',v,Hp,v)/n[:,None];response=2*np.einsum('bkd,bd->bk',dv,res)/n[:,None];F=fr-response;fid=np.sum(c*v/np.sqrt(n[:,None]),axis=1)**2
   for region,mask in [('ID',np.all((x>=1)&(x<=4),axis=1)&~np.isin(np.arange(len(x)),train)),('OOD',np.any((x<1)|(x>4),axis=1)),('anchors',np.isin(np.arange(len(x)),train))]:
    row=dict(anchors=anchors**2,side=anchors,seed=seed,region=region,success=bool(opt.success),iterations=opt.nit,seconds=time.time()-t0,E_MAE_mHa=float(np.mean(abs(en[mask]-E[mask]))*1000),E_max_mHa=float(np.max(abs(en[mask]-E[mask]))*1000),F_RMSE_mHa_bohr=float(np.sqrt(np.mean((F[mask]-F0[mask])**2))*1000),F_frozen_RMSE_mHa_bohr=float(np.sqrt(np.mean((fr[mask]-F0[mask])**2))*1000),infidelity_mean=float(np.mean(1-fid[mask])));rows.append(row)
   np.savez_compressed(P/f'dimer_n{len(train)}_s{seed}.npz',p=opt.x,E=en,F=F,F_frozen=fr,c=v/np.sqrt(n[:,None]),train_indices=train)
   print(rows[-3],flush=True)
 (P/'dimer_results.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':run()

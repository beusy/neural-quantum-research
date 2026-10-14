# Author: Don van Beusichem
import experiments as ex
import numpy as np, scipy.linalg as la
from scipy.optimize import minimize
import json,time
RPATH=ex.ROOT

def pairdesc(R):
 n=R.shape[1];ii,jj=np.triu_indices(n,1);d=R[:,ii]-R[:,jj];r=np.sqrt((d*d).sum(-1));q=1/r;J=np.zeros((len(R),len(ii),n,3))
 for a,(i,j) in enumerate(zip(ii,jj)):
  J[:,a,i,:]=-d[:,a]/r[:,a,None]**3;J[:,a,j,:]=d[:,a]/r[:,a,None]**3
 return q,J.reshape(len(R),len(ii),-1)
def blocks(x,jx,y,jy,l):
 delta=x[:,None,:]-y[None,:,:];k=np.exp(-np.sum(delta*delta,axis=-1)/(2*l*l))
 ax=np.einsum('nmd,ndc->nmc',delta,jx);ay=np.einsum('nmd,mdc->nmc',delta,jy)
 ef=-k[:,:,None]*ay/l**2;fe=k[:,:,None]*ax/l**2
 ff=k[:,:,None,None]*(np.einsum('ndc,mde->nmce',jx,jy)/l**2-ax[:,:,:,None]*ay[:,:,None,:]/l**4)
 n,m=k.shape;c=jx.shape[-1]
 return np.block([[k,ef.reshape(n,m*c)],[fe.transpose(0,2,1).reshape(n*c,m),ff.transpose(0,2,1,3).reshape(n*c,m*c)]])
def kernel_ethanol():
 data=np.load(RPATH/'data/rmd17_ethanol.npz');sp=np.load(RPATH/'splits.npz');ids=np.r_[sp['train'][:50],sp['validation'],sp['test']];R=data['coords'][ids];E=data['energies'][ids];F=data['forces'][ids];q,j=pairdesc(R);out=[]
 for n in [25,50]:
  for mode in ['E','EF']:
   cand=[]
   for l in [.25,.5,1.,2.,4.,8.]:
    K=blocks(q[:n],j[:n],q[:n],j[:n],l)
    for reg in [1e-8,1e-5,1e-3]:
     offset=E[:n].mean();y=E[:n]-offset
     if mode=='E':Ktr=K[:n,:n]
     else:Ktr=K;y=np.r_[y,F[:n].ravel()]
     a=la.solve(Ktr+reg*np.eye(len(Ktr)),y,assume_a='pos')
     B=blocks(q[50:150],j[50:150],q[:n],j[:n],l)
     if mode=='E':B=B[:,:n]
     pred=B@a;ep=pred[:100]+offset;fp=pred[100:].reshape(-1,9,3);score=np.mean((ep-E[50:150])**2)+np.mean((fp-F[50:150])**2)
     cand.append((score,l,reg,a,offset))
   _,l,reg,a,offset=min(cand,key=lambda x:x[0]);B=blocks(q[150:],j[150:],q[:n],j[:n],l)
   if mode=='E':B=B[:,:n]
   pred=B@a;out.append(dict(n=n,mode=mode,length=l,reg=reg,**ex.metric(E[150:],F[150:],pred[:300]+offset,pred[300:].reshape(-1,9,3))))
   print('kernel',out[-1],flush=True)
 # Independent kernel derivative check from energy finite differences.
 x=R[150:151];q0,j0=pairdesc(x);B=blocks(q0,j0,q[:n],j[:n],l);pred=B@a;h=1e-5
 plus=x.copy();minus=x.copy();plus[0,0,0]+=h;minus[0,0,0]-=h
 qp,jp=pairdesc(plus);qm,jm=pairdesc(minus)
 fd=-((blocks(qp,jp,q[:n],j[:n],l)@a)[0]-(blocks(qm,jm,q[:n],j[:n],l)@a)[0])/(2*h)
 return out,abs(fd-pred[1])

# Fully trainable one-layer scalar/vector network on synthetic H3.
# 3 radial channels, learned centres/positive widths, scalar/vector gates, output.
def net(R,p):
 n=R.shape[1];dR=R[:,None,:,:]-R[:,:,None,:];d=np.sqrt((dR*dR).sum(-1)+np.eye(n)[None]);u=dR/d[:,:,:,None]
 c=p[:3];beta=np.exp(p[3:6]);a=p[6:9];b=p[9:12];bias=p[12:15];out=p[15:18]
 g=np.exp(-beta*(d[:,:,:,None]-c)**2)*(1-np.eye(n))[None,:,:,None]
 s=g.sum(2);v=np.einsum('bijk,bijd->bikd',g,u);h=np.tanh(a*s+b*np.sum(v*v,axis=-1)+bias)
 return np.sum(h*out,axis=(1,2))+p[18]
def netef(R,p):
 e=net(R,p);f=np.empty_like(R)
 for i in range(R.shape[1]):
  for j in range(3):
   x=R.astype(complex);x[:,i,j]+=1e-25j;f[:,i,j]=-net(x,p).imag/1e-25
 return e,f

def network_h3():
 rr=np.random.default_rng(15)
 # Bounded, non-colliding triangles generated directly from sides/angle.
 n=150;r=rr.uniform(.8,1.6,n);s=rr.uniform(.8,1.6,n);theta=rr.uniform(.7,2.3,n);R=np.zeros((n,3,3));R[:,1,0]=r;R[:,2,0]=s*np.cos(theta);R[:,2,1]=s*np.sin(theta)
 def truth(R):
  d=np.sqrt(((R[:,[0,0,1]]-R[:,[1,2,2]])**2).sum(-1));return ex.morse(d)[0].sum(1)+2*np.exp(-1.2*d.sum(1))
 E=truth(R);F=np.empty_like(R)
 for i in range(3):
  for j in range(3):
   x=R.astype(complex);x[:,i,j]+=1e-25j;F[:,i,j]=-truth(x).imag/1e-25
 out=[]
 initial_output=rr.normal(0,.3,3)
 for w in [0,1]:
  p=np.r_[np.array([.7,1.2,1.9]),np.zeros(3),np.ones(3)*.2,np.ones(3)*.1,np.zeros(3),initial_output.copy(),-8.]
  def objective(p):
   e,f=netef(R[:60],p);return np.mean((e-E[:60])**2)+w*np.mean((f-F[:60])**2)+1e-7*np.sum(p*p)
  opt=minimize(objective,p,method='L-BFGS-B',bounds=[(.2,3.)]*3+[(-3,3)]*3+[(None,None)]*13,options={'maxiter':400,'ftol':1e-10,'maxfun':16000})
  ep,fp=netef(R[60:],opt.x);out.append(dict(w=w,success=bool(opt.success),iterations=opt.nit,message=str(opt.message),parameters=opt.x.tolist(),**ex.metric(E[60:],F[60:],ep,fp)))
  print('NN H3',out[-1],flush=True)
 return out
if __name__=='__main__':
 t=time.time();out,check=kernel_ethanol();nn=network_h3();(RPATH/'extensions.json').write_text(json.dumps(dict(kernel_ethanol=out,kernel_force_check=float(check),network_H3=nn,elapsed=time.time()-t),indent=2))

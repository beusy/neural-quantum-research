# Author: Don van Beusichem
"""Exact spin Hamiltonians and manually differentiated neural states."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import logsumexp
from pathlib import Path
import json
P=Path(__file__).parent

def configurations(n,sector=False):
 states=np.array([i for i in range(1<<n) if not sector or i.bit_count()==n//2],dtype=int)
 spins=2*((states[:,None]>>np.arange(n))&1)-1
 return states,spins.astype(float)

def tfim(n,h=1.):
 states,spins=configurations(n);rows=[];cols=[];values=[]
 for col,s in enumerate(states):
  rows.append(col);cols.append(col);values.append(-np.sum(spins[col]*np.roll(spins[col],-1)))
  for i in range(n):rows.append(int(s^(1<<i)));cols.append(col);values.append(-h)
 H=coo_matrix((values,(rows,cols)),shape=(len(states),len(states))).tocsr();return H,states,spins

def j1j2(n,j2=.7,marshall=True):
 states,spins=configurations(n,True);lookup={int(s):i for i,s in enumerate(states)};rows=[];cols=[];values=[]
 for col,s in enumerate(states):
  diag=0.
  for dist,J in [(1,1.),(2,j2)]:
   for i in range(n):
    j=(i+dist)%n;diag+=J*spins[col,i]*spins[col,j]/4
    if spins[col,i]!=spins[col,j]:
     rows.append(lookup[int(s^(1<<i)^(1<<j))]);cols.append(col);values.append(J/2)
  rows.append(col);cols.append(col);values.append(diag)
 H=coo_matrix((values,(rows,cols)),shape=(len(states),len(states))).tocsr()
 sign=(-1.)**np.sum(spins[:,::2]<0,axis=1)
 if marshall:H=H.multiply(sign[:,None]*sign[None,:]).tocsr()
 return H,states,spins

def ground(H,positive_even=False,n=None):
 e,c=eigh(H.toarray());v=c[:,0]
 if positive_even:
  v=np.abs(v+v[::-1]);v/=np.linalg.norm(v)
 if v[np.argmax(abs(v))]<0:v=-v
 return e,v

class RBM:
 def __init__(self,n,m):self.n=n;self.m=m;self.params=n+m+n*m;self.name='positive_RBM'
 def initial(self,seed):
  rr=np.random.default_rng(seed);return rr.normal(0,.05,self.params)
 def values(self,p,spins,jac=True):
  n,m=self.n,self.m;a=p[:n];b=p[n:n+m];W=p[n+m:].reshape(n,m);z=spins@W+b
  logv=spins@a+np.sum(np.logaddexp(z,-z),axis=1);v=np.exp(logv-logv.max());norm=np.linalg.norm(v);c=v/norm
  if not jac:return c
  t=np.tanh(z);O=np.c_[spins,t,(spins[:,:,None]*t[:,None,:]).reshape(len(spins),-1)];J=c[:,None]*(O-(c*c)@O)
  return c,J,O

class SignedMLP:
 def __init__(self,n,m):self.n=n;self.m=m;self.params=n*m+2*m+1;self.name='signed_MLP'
 def initial(self,seed):
  rr=np.random.default_rng(seed);return np.r_[rr.normal(0,.4,self.n*self.m),rr.normal(0,.3,self.m),rr.normal(0,.2,self.m),.1]
 def values(self,p,spins,jac=True):
  n,m=self.n,self.m;W=p[:n*m].reshape(n,m);b=p[n*m:n*m+m];u=p[n*m+m:-1];d=p[-1];t=np.tanh(spins@W+b);s=1-t*t;v=t@u+d;norm=np.linalg.norm(v);c=v/norm
  if not jac:return c
  Jv=np.c_[(spins[:,:,None]*s[:,None,:]*u).reshape(len(spins),-1),s*u,t,np.ones(len(spins))];J=Jv/norm-c[:,None]*(c@Jv/norm)
  O=Jv/np.where(abs(v)>1e-15,v,np.copysign(1e-15,v))[:,None]
  return c,J,O

def measurements(model,p,spins,H,target,E0):
 c,J,O=model.values(p,spins);hc=H@c;E=c@hc;res=hc-E*c;overlap=c@target;infid=1-overlap**2;S=J.T@J;w=np.linalg.eigvalsh(S);scale=max(w[-1],1e-300);norms=np.sqrt(np.diag(S));ok=norms>1e-14;Jn=J[:,ok]/norms[ok];wn=np.linalg.eigvalsh(Jn.T@Jn);rankcorr=int(np.sum(wn>max(wn[-1]*1e-8,1e-12)))
 # Euclidean projection onto the exact normalized-state tangent space.
 U,s,_=np.linalg.svd(J,full_matrices=False);keep=s>max(s[0]*1e-4,1e-6);proj=U[:,keep]@(U[:,keep].T@res);rr=np.linalg.norm(res);escape=float(np.linalg.norm(res-proj)/(rr+1e-30))
 return dict(E=float(E),E_error=float(E-E0),infidelity=float(max(infid,0)),variance=float(res@res),gradient_norm=float(np.linalg.norm(2*J.T@res)),QGT_rank=int(np.sum(w>max(scale*1e-8,1e-12))),QGT_correlation_rank=rankcorr,tangent_unrepresented_fraction=escape,tangent_unrepresented_norm=float(np.linalg.norm(res-proj)),parameters=model.params),dict(c=c,J=J,O=O,S=S,eigenvalues=w,residual=res)

def train(model,H,spins,target,E0,seed,objective='energy',maxiter=1500):
 p0=model.initial(seed);history=[];checkpoints={};iterations=[0]
 def obj(p):
  c,J,O=model.values(p,spins)
  if objective=='energy':hc=H@c;E=c@hc;return float(E),2*J.T@(hc-E*c)
  f=c@target;return float(1-f*f),-2*f*(J.T@target)
 def callback(p):
  iterations[0]+=1;i=iterations[0]
  if i in [1,5,25,100,500,1000,1500]:
   c=model.values(p,spins,False);E=c@(H@c);history.append(dict(iteration=i,E_error=float(E-E0),infidelity=float(1-(c@target)**2)));checkpoints[i]=p.copy()
 checkpoints[0]=p0.copy();opt=minimize(obj,p0,jac=True,method='L-BFGS-B',callback=callback,options={'maxiter':maxiter,'gtol':1e-9,'ftol':1e-14,'maxcor':25,'maxls':40})
 met,arrays=measurements(model,opt.x,spins,H,target,E0);met.update(seed=seed,objective=objective,success=bool(opt.success),iterations=opt.nit,status=str(opt.message),width=model.m)
 return met,opt.x,history,checkpoints

if __name__=='__main__':
 for h in [.2,1.,2.]:
  H,states,spins=tfim(8,h);e,c=ground(H,True,8);k=(2*np.arange(8)+1)*np.pi/8;analytic=-np.sum(np.sqrt(1+h*h-2*h*np.cos(k)));print('TFIM',h,e[0],analytic,e[0]-analytic)
 H,states,spins=j1j2(8,.7);e,c=ground(H);print('J1J2',e[:3],np.sum(c[c>0]**2),np.sum(c[c<0]**2))

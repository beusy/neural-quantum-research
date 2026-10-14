# Author: Don van Beusichem
"""Manual neural coefficient map and forward-tangent/backward derivatives.
No autodiff. Coefficients are REAL and SIGNED, normalized by Rayleigh quotient.
The determinant readout grows with CI dimension; no scalability claim.
"""
import numpy as np
from scipy.optimize import minimize
class CoefficientNet:
 def __init__(self,dim,width=24,frequency=0.):
  self.dim=dim;self.width=width;self.frequency=frequency
 def unpack(self,p):
  k=self.width;d=self.dim;return p[:k],p[k:2*k],p[2*k:2*k+k*d].reshape(k,d),p[2*k+k*d:]
 def initial(self,seed,baseline):
  rr=np.random.default_rng(seed);k=self.width
  return np.r_[rr.normal(0,.7,k),rr.normal(0,.5,k),rr.normal(0,.02,k*self.dim),baseline]
 def forward(self,x,p):
  a,b,W,c=self.unpack(p);z=x[:,None]*a+b;t=np.tanh(z);s=1-t*t
  v=t@W+c;dv=(s*a)@W
  return v,dv,(x,a,t,s,W)
 def backward(self,g,h,cache):
  x,a,t,s,W=cache;gt=g@W.T;ht=h@W.T
  ga=np.sum(gt*s*x[:,None]+ht*s*(1-2*t*a*x[:,None]),axis=0)
  gb=np.sum(gt*s-2*ht*s*t*a,axis=0)
  gW=t.T@g+(s*a).T@h;gc=g.sum(0)
  return np.r_[ga,gb,gW.ravel(),gc]

def quantities(net,p,r,H,Hprime,center=3.,scale=2.):
 x=(r-center)/scale;v,dv,_=net.forward(x,p);dv/=scale;n=np.sum(v*v,axis=1);Hv=np.einsum('bij,bj->bi',H,v);E=np.sum(v*Hv,axis=1)/n;res=Hv-E[:,None]*v
 frozen=np.einsum('bi,bij,bj->b',v,Hprime,v)/n;response=2*np.sum(dv*res,axis=1)/n;total=frozen+response
 c=v/np.sqrt(n[:,None]);dc=(dv-v*(np.sum(v*dv,axis=1)/n)[:,None])/np.sqrt(n[:,None])
 return dict(E=E,F=-total,F_frozen=-frozen,response=response,residual=np.linalg.norm(res,axis=1)/np.sqrt(n),speed=np.linalg.norm(dc,axis=1),c=c,dc=dc)

def fit(net,r,H,seed,baseline,penalty=0.,maxiter=2500,center=3.,scale=2.,initial=None):
 p0=net.initial(seed,baseline) if initial is None else initial.copy();x=(r-center)/scale;B=len(r)
 def objective(p):
  v,dv,cache=net.forward(x,p);dv=dv/scale;n=np.sum(v*v,axis=1);Hv=np.einsum('bij,bj->bi',H,v);E=np.sum(v*Hv,axis=1)/n;res=Hv-E[:,None]*v
  gradv=2*res/n[:,None];gradd=np.zeros_like(v);C=2*np.sum(dv*res,axis=1)/n
  if penalty:
   Hd=np.einsum('bij,bj->bi',H,dv)
   cp=2*(Hd-E[:,None]*dv)/n[:,None]-4*np.sum(dv*v,axis=1)[:,None]*res/n[:,None]**2-2*C[:,None]*v/n[:,None]
   cd=2*res/n[:,None];gradv+=2*penalty*C[:,None]*cp;gradd=2*penalty*C[:,None]*cd
  # Tiny weight penalty breaks unconstrained coefficient scaling.
  loss=np.mean(E+penalty*C*C)+1e-10*np.sum(p*p)
  grad=net.backward(gradv/B,gradd/B/scale,cache)+2e-10*p
  return loss,grad
 opt=minimize(objective,p0,jac=True,method='L-BFGS-B',options={'maxiter':maxiter,'maxls':40,'ftol':1e-13,'gtol':1e-8,'maxcor':30})
 return opt,objective

# Author: Don van Beusichem
"""Hydrogen STO-3G integrals, fermionic Hamiltonian and RHF from scratch.
Atomic units throughout. PySCF is optional and used ONLY by validate.py.
"""
import numpy as np
from scipy.special import erf
from scipy.linalg import eigh
from itertools import combinations
EXP=np.array([3.425250914, .6239137298, .1688554040])
COEF=np.array([.1543289673,.5353281423,.4446345422])
NORM=(2*EXP/np.pi)**.75
C=COEF*NORM

def boys0(t):
 t=np.asarray(t);return np.where(t<1e-9,1-t/3+t*t/10,np.sqrt(np.pi)/2*erf(np.sqrt(np.maximum(t,1e-300)))/np.sqrt(np.maximum(t,1e-300)))

def integrals(R):
 R=np.asarray(R);m=len(R);a=EXP[:,None];b=EXP[None,:];p=a+b;mu=a*b/p;cc=C[:,None]*C[None,:]
 S=np.zeros((m,m));T=S.copy();V=S.copy();pairs={}
 for i in range(m):
  for j in range(m):
   r2=np.sum((R[i]-R[j])**2);P=(a[:,:,None]*R[i]+b[:,:,None]*R[j])/p[:,:,None];K=np.exp(-mu*r2)
   S[i,j]=np.sum(cc*K*(np.pi/p)**1.5)
   T[i,j]=np.sum(cc*K*(np.pi/p)**1.5*mu*(3-2*mu*r2))
   for center in R:
    V[i,j]-=np.sum(cc*K*2*np.pi/p*boys0(p*np.sum((P-center)**2,axis=-1)))
   pairs[i,j]=(p.ravel(),P.reshape(-1,3),(cc*K).ravel())
 eri=np.empty((m,m,m,m))
 for i in range(m):
  for j in range(m):
   pp,PP,ww=pairs[i,j]
   for k in range(m):
    for l in range(m):
     qq,QQ,vv=pairs[k,l];den=pp[:,None]+qq[None,:];t=pp[:,None]*qq[None,:]/den*np.sum((PP[:,None,:]-QQ[None,:,:])**2,axis=-1)
     eri[i,j,k,l]=np.sum(ww[:,None]*vv[None,:]*2*np.pi**2.5/(pp[:,None]*qq[None,:]*np.sqrt(den))*boys0(t))
 enuc=sum(1/np.linalg.norm(R[i]-R[j]) for i in range(m) for j in range(i))
 return S,T+V,eri,enuc

def lowdin(S):
 w,v=eigh(S);assert min(w)>1e-7
 return (v/np.sqrt(w))@v.T

def transform(h,eri,X):
 return X.T@h@X,np.einsum('ap,bq,cr,ds,abcd->pqrs',X,X,X,X,eri,optimize=True)

def basis(m):
 occ=list(combinations(range(m),m//2));bits=[sum(1<<i for i in a)+sum(1<<(j+m) for j in b) for a in occ for b in occ]
 return np.array(bits,dtype=int),occ

def act(bit,ops):
 sign=1
 for create,i in ops:
  occupied=(bit>>i)&1
  if occupied==create:return None
  if (bit & ((1<<i)-1)).bit_count()%2:sign=-sign
  bit^=1<<i
 return bit,sign
_CACHE={}
def operators(m):
 if m in _CACHE:return _CACHE[m]
 bits,occ=basis(m);lookup={b:i for i,b in enumerate(bits)};D=len(bits);one=[];two=[]
 for p in range(m):
  for q in range(m):
   for sigma in range(2):
    for col,bit in enumerate(bits):
     a=act(int(bit),[(False,q+sigma*m),(True,p+sigma*m)])
     if a is not None:one.append((lookup[a[0]],col,p,q,a[1]))
 for p in range(m):
  for q in range(m):
   for rr in range(m):
    for s in range(m):
     for sigma in range(2):
      for tau in range(2):
       for col,bit in enumerate(bits):
        a=act(int(bit),[(False,q+sigma*m),(False,s+tau*m),(True,rr+tau*m),(True,p+sigma*m)])
        if a is not None:two.append((lookup[a[0]],col,p,q,rr,s,.5*a[1]))
 out=(np.array(one),np.array(two),D);_CACHE[m]=out;return out

def hamiltonian(h,eri,enuc=0.):
 one,two,D=operators(len(h));H=np.zeros((D,D));a=one.astype(int);np.add.at(H,(a[:,0],a[:,1]),one[:,-1]*h[a[:,2],a[:,3]])
 a=two.astype(int);np.add.at(H,(a[:,0],a[:,1]),two[:,-1]*eri[a[:,2],a[:,3],a[:,4],a[:,5]])
 H+=enuc*np.eye(D);assert np.max(abs(H-H.T))<1e-9;return H

def rhf(S,h,eri,enuc):
 X=lowdin(S);nocc=len(S)//2;eps,c=eigh(X.T@h@X);c=X@c;P=2*c[:,:nocc]@c[:,:nocc].T;fs=[];errs=[]
 def fock(P):return h+np.einsum('rs,pqrs->pq',P,eri)-.5*np.einsum('rs,prqs->pq',P,eri)
 for it in range(500):
  F=fock(P);err=(X.T@(F@P@S-S@P@F)@X).ravel();fs.append(F.copy());errs.append(err.copy())
  if len(fs)>8:fs.pop(0);errs.pop(0)
  if len(fs)>1 and np.linalg.norm(err)>1e-11:
   k=len(fs);B=np.empty((k+1,k+1));B[:k,:k]=np.array([[a@b for b in errs] for a in errs]);B[-1,:]=B[:,-1]=-1;B[-1,-1]=0;rhs=np.r_[np.zeros(k),-1.]
   try:
    co=np.linalg.solve(B,rhs)[:k]
    if np.max(abs(co))<1e5:F=np.einsum('i,ijk->jk',co,np.array(fs))
   except np.linalg.LinAlgError:pass
  eps,c=eigh(X.T@F@X);c=X@c;Pn=2*c[:,:nocc]@c[:,:nocc].T
  E=.5*np.sum(Pn*(h+fock(Pn)))+enuc
  if np.max(abs(Pn-P))<1e-9:return float(E),True,it+1
  P=Pn if it>2 else .7*Pn+.3*P
 return float(E),False,500

def geometry(n,r,kind='chain'):
 if kind=='chain':return np.c_[np.arange(n)*r,np.zeros((n,2))]
 if kind=='dimer':
  a,b=r;return np.c_[np.array([0,a,a+b,2*a+b]),np.zeros((4,2))]
 raise ValueError(kind)

def getH(n,r,kind='chain'):
 R=geometry(n,r,kind);S,h,g,en=integrals(R);X=lowdin(S);hh,gg=transform(h,g,X);return hamiltonian(hh,gg,en)

if __name__=='__main__':
 for n in [2,4]:
  R=geometry(n,1.4);S,h,g,en=integrals(R);hh,gg=transform(h,g,lowdin(S));H=hamiltonian(hh,gg,en);print(n,H.shape,eigh(H)[0][:3],rhf(S,h,g,en))

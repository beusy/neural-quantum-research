# Author: Don van Beusichem
"""Reproducible PES experiments: NumPy/SciPy only. Run with BLAS threads=1."""
import numpy as np, scipy.linalg as la
from scipy.optimize import minimize
from scipy.stats import spearmanr
import json, hashlib, time, pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).parent
rng=np.random.default_rng(20261014)
results={}
def metric(e,f,ep,fp):
 return dict(E_RMSE=float(np.sqrt(np.mean((e-ep)**2))),F_RMSE=float(np.sqrt(np.mean((f-fp)**2))),E_MAE=float(np.mean(abs(e-ep))),F_MAE=float(np.mean(abs(f-fp))))
def morse(r):
 q=np.exp(-1.8*(r-.74));return 4.75*(q*q-2*q),17.1*(q-q*q)
def basis(r,kind):
 if kind=='poly':
  x=(r-1.1)/.7; P=np.array([x**k for k in range(10)]).T; D=np.array([np.zeros_like(x)]+[k*x**(k-1)/.7 for k in range(1,10)]).T
 else:
  c=np.linspace(.4,2.4,30);d=r[:,None]-c;P=np.exp(-d*d/(2*.15**2));D=-d/.15**2*P
  P=np.c_[np.ones(len(r)),P];D=np.c_[np.zeros(len(r)),D]
 return P,D

def linear_fit(P,D,e,f,w,reg=1e-8):
 # Objective mean energy error^2 + w * mean force-component error^2.
 A=np.vstack((P/np.sqrt(len(e)),np.sqrt(w)*D.reshape(-1,P.shape[-1])/np.sqrt(f.size)))
 y=np.r_[e/np.sqrt(len(e)),np.sqrt(w)*f.ravel()/np.sqrt(f.size)]
 return la.lstsq(np.vstack((A,np.sqrt(reg)*np.eye(A.shape[1]))),np.r_[y,np.zeros(A.shape[1])])[0]

def nn(r,p):
 h=24;a,b,c=p[:h],p[h:2*h],p[2*h:3*h];d=p[-1];x=(r-1.1)/.7
 t=np.tanh(x[:,None]*a+b);s=1-t*t
 E=t@c+d;g=(s*a)@c/.7
 JE=np.c_[s*c*x[:,None],s*c,t,np.ones(len(r))]
 Jg=np.c_[c*s*(1-2*t*a*x[:,None])/.7,-2*c*s*t*a/.7,s*a/.7,np.zeros(len(r))]
 return E,g,JE,Jg

def train_nn(r,w,seed,hybrid=False):
 rr=np.random.default_rng(seed);p=np.r_[rr.normal(size=24),rr.normal(size=24),rr.normal(0,.1,24),0.]
 e,g=morse(r)
 # Deliberately imperfect, fixed prior; no fitting to test data.
 def prior(x):
  q=np.exp(-1.5*(x-.8));return 4*(q*q-2*q),12*(q-q*q)
 if hybrid:
  be,bg=prior(r);e=e-be;g=g-bg
 def objective(p):
  ep,gp,je,jg=nn(r,p);de=ep-e;dg=gp-g
  return np.mean(de*de)+w*np.mean(dg*dg)+1e-7*np.sum(p*p),2*(je.T@de+w*jg.T@dg)/len(r)+2e-7*p
 opt=minimize(objective,p,jac=True,method='L-BFGS-B',options={'maxiter':1500,'ftol':1e-12,'gtol':1e-8})
 def pred(x):
  e,g,*_=nn(x,opt.x)
  if hybrid:
   be,bg=prior(x);e+=be;g+=bg
  return e,g
 return pred,dict(success=bool(opt.success),iterations=opt.nit,loss=float(opt.fun))

def kernel(x,y,l=.35):
 d=x[:,None]-y[None,:];k=np.exp(-d*d/(2*l*l));return k,d/l**2*k,-d/l**2*k,(1/l**2-d*d/l**4)*k

def gpfit(x,mode):
 e,g=morse(x);k,ky,kx,kxy=kernel(x,x)
 if mode=='E':K=k;y=e
 else:K=np.block([[k,ky],[kx,kxy]]);y=np.r_[e,g]
 L=la.cho_factor(K+1e-8*np.eye(len(K)));a=la.cho_solve(L,y)
 def pred(z):
  k,ky,kx,kxy=kernel(z,x)
  if mode=='E':return k@a,kx@a
  return np.c_[k,ky]@a,np.c_[kx,kxy]@a
 return pred

def h2():
 records=[];curve={};grid=np.linspace(.35,3.5,700);e,g=morse(grid)
 for n in [8,16,32]:
  for seed in [0,1,2]:
   tr=np.random.default_rng(seed).uniform(.55,1.6,n)
   for kind in ['poly','rbf','kernel']:
    for w in [0,1]:
     if kind=='kernel':pred=gpfit(tr,'E' if w==0 else 'EF')
     else:
      P,D=basis(tr,kind);ee,gg=morse(tr);a=linear_fit(P,D,ee,gg,w)
      def pred(x):
       P,D=basis(x,kind);return P@a,D@a
     ep,gp=pred(grid)
     for name,mask in [('ID',(grid>=.55)&(grid<=1.6)),('short',grid<.55),('long',grid>1.6)]:
      records.append(dict(model=kind,n=n,seed=seed,w=w,region=name,**metric(e[mask],g[mask],ep[mask],gp[mask])))
     if n==16 and seed==0:curve[kind+str(w)]=(ep,gp)
 # Independent network repeats.
 for hybrid in [False,True]:
  for w in [0,1]:
   for seed in [0,1,2]:
    tr=np.random.default_rng(seed).uniform(.55,1.6,32);pred,status=train_nn(tr,w,seed,hybrid);ep,gp=pred(grid)
    for name,mask in [('ID',(grid>=.55)&(grid<=1.6)),('short',grid<.55),('long',grid>1.6)]:
     records.append(dict(model='hybrid' if hybrid else 'MLP',n=32,seed=seed,w=w,region=name,optimizer=status,**metric(e[mask],g[mask],ep[mask],gp[mask])))
    if seed==0:curve[('hybrid' if hybrid else 'MLP')+str(w)]=(ep,gp)
 fig,ax=plt.subplots(1,2,figsize=(11,4))
 for a,y in zip(ax,[e,g]):a.plot(grid,y,'k',label='Morse reference');a.axvspan(.55,1.6,color='gray',alpha=.12)
 for key in ['kernel1','MLP1','hybrid1']:
  ax[0].plot(grid,curve[key][0],label=key);ax[1].plot(grid,curve[key][1],label=key)
 ax[0].set(xlabel='r [Å]',ylabel='E [eV]',ylim=(-6,12));ax[1].set(xlabel='r [Å]',ylabel='dE/dr [eV/Å]',ylim=(-70,12))
 for a in ax:a.legend();a.grid(alpha=.2)
 fig.tight_layout();fig.savefig(ROOT/'figures/h2.png',dpi=180);plt.close(fig)
 results['h2']=records
 # MD: radial reduced mass in declared dimensionless dynamics (not fs).
 pred=gpfit(np.linspace(.55,1.6,32),'EF');md=[]
 for dt in [.02,.01,.005]:
  x=1.2;v=0.;t0=pred(np.array([x]))[0][0];vals=[];refvals=[]
  for step in range(round(20/dt)):
   force=-pred(np.array([x]))[1][0];v+=.5*dt*force;x+=dt*v;v-=.5*dt*pred(np.array([x]))[1][0]
   vals.append(.5*v*v+pred(np.array([x]))[0][0]);refvals.append(.5*v*v+morse(np.array([x]))[0][0])
  md.append(dict(dt=dt,max_learned_energy_deviation=float(np.max(abs(np.array(vals)-t0))),max_reference_energy_deviation=float(np.max(abs(np.array(refvals)-morse(np.array([1.2]))[0][0])))))
 results['MD']=md
 # Shared blind spot: bootstrap identical RBF model, beyond all centres.
 ensemble=[]
 for seed in range(12):
  r=np.random.default_rng(seed).uniform(.55,1.6,16);P,D=basis(r,'rbf');ee,gg=morse(r);a=linear_fit(P,D,ee,gg,1);ensemble.append(basis(grid,'rbf')[0]@a)
 ens=np.array(ensemble);std=ens.std(0);err=abs(ens.mean(0)-e)
 results['uncertainty']={'spearman_ID':float(spearmanr(std[(grid>=.55)&(grid<=1.6)],err[(grid>=.55)&(grid<=1.6)]).statistic),'at_3_5_A':{'std_eV':float(std[-1]),'abs_error_eV':float(err[-1])}}
 print('H2 complete',flush=True)

# Smooth features with scalar densities and polar-vector channels. All contractions bilinear.
def features(R,Z,vector=True):
 B,N,_=R.shape;ds=R[:,None,:,:]-R[:,:,None,:];mask=1-np.eye(N)
 d=np.sqrt(np.sum(ds*ds,axis=-1)+np.eye(N)[None,:,:]);unit=ds/d[:,:,:,None]
 centers=np.linspace(.6,4.,8);rad=np.exp(-((d[:,:,:,None]-centers)/.6)**2)*mask[None,:,:,None]
 species=sorted(set(Z));S=[];V=[]
 for z in species:
  sel=(Z==z)[None,None,:,None];S.append(np.sum(rad*sel,axis=2));V.append(np.einsum('bijk,bijd->bikd',rad*sel,unit))
 S=np.concatenate(S,axis=-1);V=np.concatenate(V,axis=2)
 rows=[np.ones(B,dtype=R.dtype)]
 for z in species:
  sel=(Z==z);s=S[:,sel];rows.extend(np.sum(s,axis=1).T)
  for a in range(S.shape[-1]):
   for b in range(a,S.shape[-1]):rows.append(np.sum(s[:,:,a]*s[:,:,b],axis=1))
  if vector:
   v=V[:,sel]
   # same-radial-channel and adjacent-channel vector contractions, deliberately compact
   for a in range(V.shape[2]):
    for b in [a]+([a+1] if a+1<V.shape[2] else []):rows.append(np.sum(v[:,:,a,:]*v[:,:,b,:],axis=(1,2)))
 return np.array(rows).T

def feature_derivatives(R,Z,vector=True):
 P=features(R,Z,vector);D=np.empty((len(R),R.shape[1],3,P.shape[1]));h=1e-25
 for i in range(R.shape[1]):
  for j in range(3):
   x=R.astype(complex);x[:,i,j]+=h*1j;D[:,i,j]=-features(x,Z,vector).imag/h
 return P,D

def fit_scaled(P,D,e,f,w,reg):
 mean=P.mean(0);scale=P.std(0);scale[scale<1e-8]=1;mean[0]=0;scale[0]=1
 pp=(P-mean)/scale;dd=D/scale;offset=e.mean();a=linear_fit(pp,dd,e-offset,f,w,reg)
 return a,mean,scale,offset

def apply(P,D,model):
 a,m,s,o=model;return (P-m)/s@a+o,np.einsum('...k,k->...',D/s,a)

def h3():
 rr=np.random.default_rng(8);R=rr.normal(size=(260,3,3))*.55;R[:,1,0]+=1.4;R[:,2,1]+=1.4;Z=np.ones(3,int)
 def truth(R):
  d=np.sqrt(np.sum((R[:,[0,0,1]]-R[:,[1,2,2]])**2,axis=-1));e=morse(d)[0].sum(1);q=np.exp(-1.2*d);return e+2*np.prod(q,axis=1)
 E=truth(R);F=np.empty_like(R)
 for i in range(3):
  for j in range(3):
   x=R.astype(complex);x[:,i,j]+=1e-25j;F[:,i,j]=-truth(x).imag/1e-25
 out=[]
 for v in [False,True]:
  P,D=feature_derivatives(R,Z,v);model=fit_scaled(P[:100],D[:100],E[:100],F[:100],1,1e-5);ep,fp=apply(P[100:],D[100:],model)
  out.append(dict(vector=v,**metric(E[100:],F[100:],ep,fp)))
 results['H3_synthetic']=out;print('H3 complete',flush=True)

def ethanol():
 data=np.load(ROOT/'data/rmd17_ethanol.npz');R=data['coords'];E=data['energies'];F=data['forces'];Z=data['nuclear_charges'];N=len(R)
 # Hold out fixed random test and validation sets. Nested train subsets.
 ids=np.random.default_rng(42).permutation(N);train=ids[:200];val=ids[200:300];test=ids[300:600];sel=np.r_[train,val,test]
 np.savez(ROOT/'splits.npz',train=train,validation=val,test=test)
 R,E,F=R[sel],E[sel],F[sel];out=[];cache={}
 for v in [False,True]:
  start=time.time();P,D=feature_derivatives(R,Z,v);cache[v]=(P,D);print('ethanol features',v,P.shape,time.time()-start,flush=True)
  for n in [25,100,200]:
   for w in [0,1,100]:
    candidates=[]
    for reg in [1e-6,1e-3,1.]:
     mod=fit_scaled(P[:n],D[:n],E[:n],F[:n],w,reg);ep,fp=apply(P[200:300],D[200:300],mod)
     score=np.mean((ep-E[200:300])**2)+np.mean((fp-F[200:300])**2);candidates.append((score,reg,mod))
    _,reg,mod=min(candidates,key=lambda x:x[0]);ep,fp=apply(P[300:],D[300:],mod)
    out.append(dict(vector=v,n=n,w=w,reg=reg,features=P.shape[1],**metric(E[300:],F[300:],ep,fp)))
    print(out[-1],flush=True)
 results['ethanol']=out
 # Symmetry and differential verification on selected learned vector model.
 P,D=cache[True];mod=fit_scaled(P[:200],D[:200],E[:200],F[:200],1,1e-3);x=R[:3];p,d=feature_derivatives(x,Z);e,f=apply(p,d,mod)
 Q=np.linalg.qr(np.random.default_rng(3).normal(size=(3,3)))[0];xp=x@Q+np.array([2.,-3.,1.]);pp,dd=feature_derivatives(xp,Z);ee,ff=apply(pp,dd,mod)
 perm=np.arange(len(Z));hyd=np.where(Z==1)[0];perm[hyd]=hyd[::-1];pp,dd=feature_derivatives(x[:,perm],Z[perm]);pe,pf=apply(pp,dd,mod)
 h=1e-5;xp=x.copy();xm=x.copy();xp[0,0,0]+=h;xm[0,0,0]-=h
 a,m,s,o=mod;fd=-(((features(xp,Z)-m)/s@a+o)[0]-((features(xm,Z)-m)/s@a+o)[0])/(2*h)
 results['checks']={'rotation_translation_energy_max':float(max(abs(ee-e))),'rotation_force_max':float(np.max(abs(ff-f@Q))),'permutation_energy_max':float(max(abs(pe-e))),'permutation_force_max':float(np.max(abs(pf-f[:,perm]))),'net_force_max':float(np.max(abs(f.sum(1)))),'torque_max':float(np.max(abs(np.cross(x,f).sum(1)))),'finite_difference_force_error':float(abs(fd-f[0,0,0]))}
 results['data']={'file_sha256':hashlib.sha256((ROOT/'data/rmd17_ethanol.npz').read_bytes()).hexdigest(),'N':N,'Z':Z.tolist(),'split_seed':42}
 fig,ax=plt.subplots(1,2,figsize=(10,4))
 for v in [False,True]:
  for w in [0,1]:
   rows=[r for r in out if r['vector']==v and r['w']==w];label=f"{'scalar+vector' if v else 'scalar'}; w={w}"
   for a,key in zip(ax,['E_MAE','F_MAE']):a.loglog([r['n'] for r in rows],[r[key] for r in rows],'o-',label=label)
 for a,y in zip(ax,['Energy MAE [kcal/mol]','Force MAE [kcal/mol/Å]']):a.set(xlabel='Training geometries',ylabel=y);a.legend(fontsize=7);a.grid(alpha=.2)
 fig.tight_layout();fig.savefig(ROOT/'figures/ethanol.png',dpi=180);plt.close(fig)

if __name__=='__main__':
 t=time.time();h2();h3();ethanol();results['elapsed_seconds']=time.time()-t
 (ROOT/'results.json').write_text(json.dumps(results,indent=2));print('Done',results['elapsed_seconds'],results['checks'])

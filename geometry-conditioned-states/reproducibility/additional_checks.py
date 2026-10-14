# Author: Don van Beusichem
import chemistry as ch
import numpy as np,json
from scipy.linalg import eigh
from scipy.interpolate import CubicSpline
from train import data
from neural import CoefficientNet,quantities
from pathlib import Path
P=Path(__file__).parent
out={};baselines=[];boundrows=[]
for n in [2,4,6]:
 a,Hp=data(n);r,H,E,c,gap=[a[k] for k in ['r','H','E','c','gap']];F0=-np.einsum('bi,bij,bj->b',c,Hp,c);nodes=np.linspace(1,5,9);idx=np.array([np.argmin(abs(r-x)) for x in nodes]);mask=(r>=1)&(r<=5)&~np.isin(np.arange(len(r)),idx)
 ref=c[idx].copy()
 for i in range(1,len(ref)):
  if ref[i]@ref[i-1]<0:ref[i]*=-1
 cs=CubicSpline(nodes,ref);v=cs(r);dv=cs(r,1);norm=(v*v).sum(1);Hv=np.einsum('bij,bj->bi',H,v);ep=(v*Hv).sum(1)/norm;resp=2*np.sum(dv*(Hv-ep[:,None]*v),axis=1)/norm;fp=-np.einsum('bi,bij,bj->b',v,Hp,v)/norm-resp
 for name,ee,ff in [('FCI coefficient spline',ep,fp),('FCI energy spline',CubicSpline(nodes,E[idx])(r),-CubicSpline(nodes,E[idx])(r,1)),('RHF',a['RHF'],-CubicSpline(r,a['RHF'])(r,1))]:
  baselines.append(dict(system=f'H{n}',model=name,E_MAE_mHa=float(np.mean(abs(ee[mask]-E[mask]))*1000),F_RMSE_mHa_bohr=float(np.sqrt(np.mean((ff[mask]-F0[mask])**2))*1000),min_energy_error_Ha=float(np.min(ee[mask]-E[mask]))))
 # Force derivative convergence and energy/force consistency of saved neural parameters.
 fname=P/(f'H{n}_n9_w{12 if n==2 else 24}_p0_s0.npz');model=np.load(fname);p=model['parameters'];net=CoefficientNet(H.shape[1],12 if n==2 else 24);q=quantities(net,p,r,H,Hp);errs=[];hpchecks=[]
 for x in [1.3,2.7,4.6]:
  h=1e-4;D=(ch.getH(n,x+h)-ch.getH(n,x-h))/(2*h);D2=(ch.getH(n,x+h/2)-ch.getH(n,x-h/2))/h;hpchecks.append(np.max(abs(D-D2)))
  H0=ch.getH(n,x);qq=quantities(net,p,np.array([x]),H0[None],D[None]);hfd=2e-5;Hp1=ch.getH(n,x+hfd);Hm1=ch.getH(n,x-hfd);zp=quantities(net,p,np.array([x+hfd]),Hp1[None],D[None]);zm=quantities(net,p,np.array([x-hfd]),Hm1[None],D[None]);fd=-(zp['E'][0]-zm['E'][0])/(2*hfd);errs.append(abs(fd-qq['F'][0]))
 out[f'H{n}']={'Hamiltonian_derivative_halfstep_max':float(max(hpchecks)),'neural_force_central_difference_max':float(max(errs))}
 normHp=np.max(abs(np.linalg.eigvalsh(Hp)),axis=1);de=np.maximum(q['E']-E,0);bound=2*normHp*np.sqrt(np.minimum(1,de/gap))+2*q['residual']*q['speed'];actual=abs(q['F']-F0)
 boundrows.append(dict(system=f'H{n}',bound_RMS_mHa_bohr=float(np.sqrt(np.mean(bound[mask]**2))*1000),actual_RMS_mHa_bohr=float(np.sqrt(np.mean(actual[mask]**2))*1000),max_violation=float(np.max(actual-bound))))
 # Spin-squared: S- S+ in fixed Mz=0 subspace.
 bits,occ=ch.basis(n);lookup={int(b):i for i,b in enumerate(bits)};S2=np.zeros_like(H[0])
 for col,b in enumerate(bits):
  for i in range(n):
   for j in range(n):
    z=ch.act(int(b),[(False,j+n),(True,j),(False,i),(True,i+n)])
    if z:S2[lookup[z[0]],col]+=z[1]
 spin=np.einsum('bi,ij,bj->b',q['c'],S2,q['c']);out[f'H{n}']['S2_max_ID']=float(np.max(abs(spin[mask])));out[f'H{n}']['S2_exact_max']=float(np.max(abs(np.einsum('bi,ij,bj->b',c,S2,c))))
(P/'additional_checks.json').write_text(json.dumps(out,indent=2));(P/'baselines.json').write_text(json.dumps(baselines,indent=2));(P/'bounds.json').write_text(json.dumps(boundrows,indent=2));print(out,baselines,boundrows)

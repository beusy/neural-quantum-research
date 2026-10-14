# Author: Don van Beusichem
import chemistry as c
import numpy as np,json,time
from scipy.linalg import eigh
from pathlib import Path
P=Path(__file__).parent
for n,grid in [(2,np.linspace(.8,6.,209)),(4,np.linspace(.8,6.,209)),(6,np.linspace(1.,5.,81))]:
 t=time.time();Hs=[];Es=[];Cs=[];gaps=[];rhs=[]
 for i,r in enumerate(grid):
  R=c.geometry(n,r);S,h,g,en=c.integrals(R);hh,gg=c.transform(h,g,c.lowdin(S));H=c.hamiltonian(hh,gg,en);e,v=eigh(H);Hs.append(H);Es.append(e[0]);Cs.append(v[:,0]);gaps.append(e[1]-e[0]);rhs.append(c.rhf(S,h,g,en)[0])
  if i%40==0:print(n,i,len(grid),time.time()-t,flush=True)
 Hs=np.array(Hs);Cs=np.array(Cs)
 # Eigenvector signs aligned only for plotting / supervised reference interpolation.
 for i in range(1,len(Cs)):
  if Cs[i]@Cs[i-1]<0:Cs[i]*=-1
 np.savez_compressed(P/f'H{n}_data.npz',r=grid,H=Hs,E=Es,c=Cs,gap=gaps,RHF=rhs)
 print('complete',n,time.time()-t,flush=True)

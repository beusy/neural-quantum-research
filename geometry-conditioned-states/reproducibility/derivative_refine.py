# Author: Don van Beusichem
"""Five-point Hamiltonian derivatives for final force scoring."""
from train import data
import chemistry as c,numpy as np,json
from pathlib import Path
P=Path(__file__).parent
for n in [2,4,6]:
 a=np.load(P/f'H{n}_data.npz');hp=[];h=.001
 for x in a['r']:hp.append((-c.getH(n,x+2*h)+8*c.getH(n,x+h)-8*c.getH(n,x-h)+c.getH(n,x-2*h))/(12*h))
 np.savez_compressed(P/f'H{n}_derivatives.npz',Hp=np.array(hp),step=h,stencil='five-point')
 print('derivatives refined',n,flush=True)

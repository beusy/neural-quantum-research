# Author: Don van Beusichem
from pathlib import Path
import chemistry as c
import numpy as np,json
from pyscf import gto,scf,fci,lib
lib.num_threads(1)
# Some containers expose a host /proc mount with different process IDs.
import os
if not os.path.exists(f"/proc/{os.getpid()}/statm"):
 lib.current_memory=lambda: (0.,0.)
rows=[]
for n,r in [(2,1.4),(2,4.),(4,1.4),(4,3.)]:
 R=c.geometry(n,r);S,h,g,en=c.integrals(R);mol=gto.M(atom=[('H',tuple(x)) for x in R],unit='Bohr',basis='sto-3g',verbose=0)
 S0=mol.intor('int1e_ovlp');h0=mol.intor('int1e_kin')+mol.intor('int1e_nuc');g0=mol.intor('int2e');X=c.lowdin(S);hh,gg=c.transform(h,g,X);H=c.hamiltonian(hh,gg,en)
 efci=np.linalg.eigvalsh(H)[0];ref,_=fci.direct_spin1.kernel(hh,gg,n,n,ecore=en)
 mf=scf.RHF(mol);mf.conv_tol=1e-12;erhf=mf.kernel();ours=c.rhf(S,h,g,en)
 row={'n':n,'r_bohr':r,'S_max':float(np.max(abs(S-S0))),'h_max':float(np.max(abs(h-h0))),'eri_max':float(np.max(abs(g-g0))),'FCI_E':float(efci),'FCI_assembly_error':float(abs(efci-ref)),'RHF_E':ours[0],'RHF_error':float(abs(ours[0]-erhf)),'RHF_converged':ours[1]};rows.append(row);print(row,flush=True)
assert max(x['FCI_assembly_error'] for x in rows)<1e-9
assert max(x['h_max'] for x in rows)<1e-7
open(Path(__file__).parent/'validation.json','w').write(json.dumps(rows,indent=2))

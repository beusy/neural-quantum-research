# Author: Don van Beusichem
import chemistry as ch,numpy as np,json
from pathlib import Path
P=Path(__file__).parent
rows=[]
for n in [2,4,6]:
 f=P/f'H{n}_data.npz';old=dict(np.load(f));es=[];status=[];its=[]
 for r in old['r']:
  S,h,g,en=ch.integrals(ch.geometry(n,r));e,success,it=ch.rhf(S,h,g,en);es.append(e);status.append(success);its.append(it)
 old.update(RHF=np.array(es),RHF_converged=np.array(status),RHF_iterations=np.array(its));np.savez_compressed(f,**old);rows.append(dict(n=n,failures=int(np.sum(~np.array(status))),max_iterations=max(its)));print(rows[-1],flush=True)
(P/'RHF_status.json').write_text(json.dumps(rows,indent=2))

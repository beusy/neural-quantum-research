# Author: Don van Beusichem
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
p=Path(__file__).parent;r=json.loads((p/'results.json').read_text());d=json.loads((p/'diagnostics.json').read_text())
fig,ax=plt.subplots(1,2,figsize=(10,4))
for name in ['random','energy_variance','force_variance']:
 rows=d['active_learning'];means=[]
 for n in [4,8,16]:means.append(np.mean([x['F_RMSE'] for x in rows if x['n']==n and x['acquisition']==name]))
 ax[0].loglog([4,8,16],means,'o-',label=name.replace('_',' '))
ax[0].set(xlabel='Labeled geometries (energy + derivative)',ylabel='Radial force RMSE [eV/Å]');ax[0].legend(fontsize=8)
rows=r['MD'];ax[1].loglog([x['dt'] for x in rows],[x['max_learned_energy_deviation'] for x in rows],'o-',label='Learned Hamiltonian');ax[1].loglog([x['dt'] for x in rows],[x['max_reference_energy_deviation'] for x in rows],'o-',label='Reference Hamiltonian');ax[1].set(xlabel='Dimensionless timestep',ylabel='Maximum energy deviation [eV]');ax[1].legend(fontsize=8)
for a in ax:a.grid(alpha=.2)
fig.tight_layout();fig.savefig(p/'figures/diagnostics.png',dpi=180)

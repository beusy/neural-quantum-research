# Author: Don van Beusichem
from core import *
rows=json.loads((P/'training_results.json').read_text());out=[]
for row in rows:
 name,family,w,seed,obj=[row[k] for k in ['system','family','width','seed','objective']];a=np.load(P/f'{name}_reference.npz');model=RBM(a['spins'].shape[1],w) if family=='RBM' else SignedMLP(a['spins'].shape[1],w);p=np.load(P/f'{name}_{family}_w{w}_s{seed}_{obj}.npz')['p'];c=model.values(p,a['spins'],False);row['mean_magnetization']=float((c*c)@a['spins'].mean(1));row['spin_flip_expectation']=float(c@c[::-1]);row['gap']=float(a['E'][1]-a['E'][0]);row['gap_fidelity_bound']=float(min(1,max(row['E_error'],0)/row['gap']))
for row in rows:
 if row['objective']!='energy':continue
 witness=next(x for x in rows if x['system']==row['system'] and x['family']==row['family'] and x['width']==row['width'] and x['seed']==row['seed'] and x['objective']=='infidelity')
 best=min(x['E_error'] for x in rows if x['system']==row['system'] and x['family']==row['family'] and x['width']==row['width'])
 out.append(dict(system=row['system'],family=row['family'],width=row['width'],seed=row['seed'],energy_run_error=row['E_error'],supervised_witness_error=witness['E_error'],matched_witness_optimization_lower_bound=max(row['E_error']-witness['E_error'],0),pooled_witness_optimization_lower_bound=max(row['E_error']-best,0)))
(P/'training_results.json').write_text(json.dumps(rows,indent=2));(P/'optimization_witnesses.json').write_text(json.dumps(out,indent=2));print('updated',len(rows),'training fits')

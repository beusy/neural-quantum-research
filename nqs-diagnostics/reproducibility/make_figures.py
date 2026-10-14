# Author: Don van Beusichem
from core import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
R=json.loads((P/'training_results.json').read_text());Q=json.loads((P/'qgt_results.json').read_text());S=json.loads((P/'sampling_results.json').read_text());C=json.loads((P/'coordinate_results.json').read_text());Y=json.loads((P/'symmetry_results.json').read_text())
def save(fig,name):fig.tight_layout();fig.savefig(P/'figures'/name,dpi=200);plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(10,4))
for aa,system,title in zip(ax,['ising_ferro','ising_critical'],['Ferromagnetic Ising, h=0.2','Critical Ising, h=1']):
 for obj,marker in [('energy','o'),('infidelity','s')]:
  for w,color in [(4,'C0'),(8,'C1'),(16,'C2')]:
   q=[x for x in R if x['system']==system and x['width']==w and x['objective']==obj];aa.scatter([max(x['E_error'],1e-14) for x in q],[max(x['infidelity'],1e-14) for x in q],marker=marker,color=color,label=f'{obj}; m={w}')
 if system=='ising_ferro':aa.scatter([x['E_error'] for x in Y if x['width']==8],[x['infidelity'] for x in Y if x['width']==8],marker='*',s=100,color='k',label='Z2 projected; m=8')
 aa.set(xscale='log',yscale='log',xlabel='Total energy error [J]',ylabel='Infidelity',title=title);aa.grid(alpha=.2);aa.legend(fontsize=6)
save(fig,'energy_fidelity.png')
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
for fam in ['RBM','MLP']:
 for obj,ls in [('energy','-'),('infidelity','--')]:
  vals=[];ranks=[]
  for w in [4,8,16]:
   q=[x for x in R if x['system']=='frustrated' and x['family']==fam and x['width']==w and x['objective']==obj];vals.append(np.mean([x['infidelity'] for x in q]));ranks.append(np.mean([x['QGT_rank'] for x in q]))
  ax[0].semilogy([4,8,16],vals,'o'+ls,label=f'{fam}; {obj}');ax[1].plot([4,8,16],ranks,'o'+ls,label=f'{fam}; {obj}')
ax[0].axhline(.5,color='k',ls=':',label='Nonnegative-state lower bound');ax[0].set(xlabel='Hidden width',ylabel='Mean infidelity');ax[1].set(xlabel='Hidden width',ylabel='Mean numerical QGT rank')
for aa in ax:aa.grid(alpha=.2);aa.legend(fontsize=7)
save(fig,'sign_restriction.png')
fig,ax=plt.subplots(1,2,figsize=(11,3.8))
for method,color in [('local_up','C0'),('local_down','C1'),('global_flip','C2')]:
 a=np.load(P/f'samples_{method}.npz');ax[0].plot(np.arange(0,len(a['magnetization']),8),a['magnetization'][::8],label=method,color=color,lw=.6)
methods=['iid','local_up','local_down','global_flip']
for i,m in enumerate(methods):
 q=[x for x in S if x['method']==m];ax[1].scatter(np.ones(len(q))*i,[x['mean_magnetization'] for x in q],s=15)
ax[0].set(xlabel='Retained sweep',ylabel='Magnetization per spin',title='Representative frozen-state chains');ax[0].legend(fontsize=8);ax[1].axhline(0,color='k',ls=':');ax[1].set(xticks=range(4),xticklabels=['IID','local up','local down','global flip'],ylabel='Estimated mean magnetization',title='Twenty seeds; exact mean is zero')
for aa in ax:aa.grid(alpha=.2)
save(fig,'sampling.png')
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
for method in ['metric_only_noise','gradient_only_noise','joint_ridge','joint_diagonal_shift','split_filter']:
 q=[]
 for shots in [16,64,256,1024]:q.append(np.mean([x['linear_uphill'] for x in Q if x['checkpoint']==25 and x['shots']==shots and x['radius']==.002 and x['method']==method]))
 ax[0].semilogx([16,64,256,1024],q,'o-',label=method.replace('_',' '))
ax[0].set(xlabel='Independent Born samples',ylabel='Fraction of first-order uphill directions');ax[0].legend(fontsize=6)
for method in ['population_small_ridge','joint_small_ridge','population','joint_ridge']:
 vals=[]
 for radius in [.002,.01,.02]:vals.append(np.mean([x['finite_uphill'] for x in Q if x['checkpoint']==25 and x['shots']==256 and x['radius']==radius and x['method']==method]))
 ax[1].semilogx([.002,.01,.02],vals,'o-',label=method.replace('_',' '))
ax[1].set(xlabel='Predicted tangent displacement',ylabel='Fraction of finite steps increasing energy');ax[1].legend(fontsize=6)
for aa in ax:aa.grid(alpha=.2)
save(fig,'qgt_noise.png')
fig,ax=plt.subplots(1,2,figsize=(10,3.8));base=[x for x in C if x['method']=='isotropic_shift'];ax[0].plot([x['log10_coordinate_extent'] for x in base],[x['QGT_threshold_rank'] for x in base],'o-',label='Raw metric');ax[0].plot([x['log10_coordinate_extent'] for x in base],[x['correlation_threshold_rank'] for x in base],'s-',label='Column-normalized metric');ax[0].set(xlabel='Log10 coordinate scale extent',ylabel='Thresholded numerical rank')
for m in ['isotropic_shift','diagonal_shift']:
 q=[x for x in C if x['method']==m];ax[1].plot([x['log10_coordinate_extent'] for x in q],[x['physical_direction_cosine'] for x in q],'o-',label=m.replace('_',' '))
ax[1].set(xlabel='Log10 coordinate scale extent',ylabel='Physical direction cosine to original update')
for aa in ax:aa.grid(alpha=.2);aa.legend(fontsize=8)
save(fig,'coordinates.png')
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
for aa,system in zip(ax,['ising_ferro','ising_critical']):
 for method in ['Adam','SR_isotropic','SR_diagonal','SR_block']:
  h=json.loads((P/f'optimizer_{system}_{method}_s0_history.json').read_text());aa.semilogy([x['Hamiltonian_evaluations'] for x in h],[max(x['E_error'],1e-14) for x in h],label=method.replace('_',' '))
 aa.set(xlabel='Recorded Hamiltonian-vector evaluations',ylabel='Energy error [J]',title=system.replace('_',' '));aa.legend(fontsize=7);aa.grid(alpha=.2)
save(fig,'optimizers.png')

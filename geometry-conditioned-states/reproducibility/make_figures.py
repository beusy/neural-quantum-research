# Author: Don van Beusichem
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
P=Path(__file__).parent;(P/'figures').mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):fig.tight_layout();fig.savefig(P/'figures'/name,dpi=200);plt.close(fig)
def load(n):
 a=np.load(P/f'H{n}_data.npz');hp=np.load(P/f'H{n}_derivatives.npz')['Hp'];return a,hp
# Physics and fidelity.
a,hp=load(4);r=a['r'];f=np.load(P/'H4_n9_w24_p0_s0.npz');F0=-np.einsum('bi,bij,bj->b',a['c'],hp,a['c'])
fig,ax=plt.subplots(1,3,figsize=(13,3.7));ax[0].plot(r,a['E'],'k',label='FCI');ax[0].plot(r,f['E'],'--',label='Neural');valid=a['RHF_converged'];ax[0].plot(r[valid],a['RHF'][valid],':',label='RHF, converged points');ax[0].set(xlabel='Spacing r [bohr]',ylabel='Total energy [Ha]');ax[0].legend(fontsize=8)
ax[1].semilogy(r,np.maximum((f['E']-a['E'])*1000,1e-9),label='Energy error [mHa]');ax[1].semilogy(r,np.maximum(abs(f['F']-F0)*1000,1e-9),label='Total force error [mHa/bohr]');ax[1].semilogy(r,np.maximum(abs(f['F_frozen']-F0)*1000,1e-9),label='Fixed-coefficient force error');ax[1].set(xlabel='Spacing r [bohr]',ylabel='Absolute errors');ax[1].legend(fontsize=7)
o=np.load(P/'occupations_H4.npz')
for k in range(4):ax[2].plot(r,o['exact'][:,k],color=f'C{k}');ax[2].plot(r,o['neural'][:,k],'--',color=f'C{k}')
ax[2].set(xlabel='Spacing r [bohr]',ylabel='Natural occupations (solid FCI; dashed neural)',ylim=(-.05,2.05))
for x in ax:x.axvspan(1,5,color='grey',alpha=.08);x.grid(alpha=.2)
save(fig,'physics.png')
# Data efficiency.
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
for n,w in [(2,12),(4,24),(6,24)]:
 rows=json.loads((P/f'H{n}_results.json').read_text());ns=sorted(set(x['anchors'] for x in rows if x['width']==w and x['penalty']==0));es=[];fs=[]
 for a in ns:
  q=[x for x in rows if x['anchors']==a and x['width']==w and x['penalty']==0 and x['region']=='ID'];es.append(np.mean([x['E_MAE_mHa'] for x in q]));fs.append(np.mean([x['F_RMSE_mHa_bohr'] for x in q]))
 ax[0].loglog(ns,es,'o-',label=f'H{n}');ax[1].loglog(ns,fs,'o-',label=f'H{n}')
for x,y in zip(ax,['Energy MAE [mHa]','Total force RMSE [mHa/bohr]']):x.set(xlabel='Anchor geometries',ylabel=y);x.grid(alpha=.2);x.legend()
save(fig,'anchors.png')
# Gauge test curves.
a,hp=load(4)
fig,ax=plt.subplots(1,2,figsize=(10,3.8));g=np.load(P/'gauge_n9_s0.npz')
for label,z in [('Smooth chart',f),('Twisted chart',g)]:
 ax[0].semilogy(r,np.maximum((z['E']-a['E'])*1000,1e-8),label=label);ax[1].semilogy(r,np.maximum(abs(z['F']-F0)*1000,1e-8),label=label)
for x,y in zip(ax,['Energy error [mHa]','Total force error [mHa/bohr]']):x.set(xlabel='Spacing r [bohr]',ylabel=y);x.axvspan(1,5,color='grey',alpha=.08);x.legend();x.grid(alpha=.2)
save(fig,'gauge.png')
# Two-dimensional H4.
a2=np.load(P/'dimer_data.npz');z=np.load(P/'dimer_n25_s0.npz');x=a2['x'];hp2=a2['Hp'];Fref=-np.einsum('bi,bkij,bj->bk',a2['c'],hp2,a2['c']);grid=np.unique(x[:,0]);fig,ax=plt.subplots(1,3,figsize=(12,3.5))
for aa,values,title in zip(ax,[a2['E'],np.maximum((z['E']-a2['E'])*1000,1e-8),np.sqrt(np.mean((z['F']-Fref)**2,axis=1))*1000],['FCI energy [Ha]','Neural energy error [mHa]','Force-vector RMS error [mHa/bohr]']):
 im=aa.pcolormesh(grid,grid,values.reshape(len(grid),len(grid)).T,shading='nearest');fig.colorbar(im,ax=aa);aa.set(xlabel='Outer spacing a [bohr]',ylabel='Middle spacing b [bohr]',title=title,aspect='equal');aa.scatter(x[z['train_indices'],0],x[z['train_indices'],1],s=8,c='white',edgecolors='black',linewidth=.3)
save(fig,'surface.png')
# Control and collocation negative/positive findings.
control=json.loads((P/'counterexample.json').read_text());ad=json.loads((P/'adaptive_results.json').read_text());fig,ax=plt.subplots(1,2,figsize=(10,3.8));ax[0].loglog([x['omega'] for x in control],[x['E_max_mHa'] for x in control],'o-',label='Maximum energy error [mHa]');ax[0].loglog([x['omega'] for x in control],[x['F_RMSE_mHa_bohr'] for x in control],'s-',label='Force RMSE [mHa/bohr]');ax[0].set(xlabel='Mixing frequency [bohr⁻¹]',ylabel='Error');ax[0].legend(fontsize=8)
methods=['uniform','random','residual','response'];means=[np.mean([x['F_RMSE_mHa_bohr'] for x in ad if x['method']==m]) for m in methods];std=[np.std([x['F_RMSE_mHa_bohr'] for x in ad if x['method']==m]) for m in methods];ax[1].bar(methods,means,yerr=std,capsize=3);ax[1].set(ylabel='Total force RMSE [mHa/bohr]',title='Nine anchors; three-stage optimization')
for aa in ax:aa.grid(alpha=.2)
save(fig,'control.png')

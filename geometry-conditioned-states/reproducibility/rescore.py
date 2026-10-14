# Author: Don van Beusichem
"""Re-evaluate saved parameters on genuinely untrained query geometries."""
from pathlib import Path
import numpy as np,json
from train import data,summarize
from neural import CoefficientNet
P=Path(__file__).parent
for n in [2,4,6]:
 f=P/f'H{n}_results.json'
 if not f.exists():continue
 rows=json.loads(f.read_text());models={}
 for row in rows:
  key=(row['anchors'],row['width'],row['penalty'],row['seed']);models[key]={k:v for k,v in row.items() if k not in ['system','region','count','E_MAE_mHa','E_max_mHa','E_min_Ha','F_RMSE_mHa_bohr','F_frozen_RMSE_mHa_bohr','response_RMS_mHa_bohr','infidelity_mean','residual_RMS_Ha']}
 a,Hp=data(n);r,H,E,c,gap=[a[k] for k in ['r','H','E','c','gap']];new=[]
 for (anc,w,pen,seed),mod in models.items():
  v=np.load(P/f'H{n}_n{anc}_w{w}_p{pen:g}_s{seed}.npz');p=v['parameters'];rr,q=summarize(n,r,H,Hp,E,c,gap,mod,CoefficientNet(H.shape[1],w),p);new+=rr
 f.write_text(json.dumps(new,indent=2))

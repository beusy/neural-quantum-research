# Author: Don van Beusichem
from core import *
rows=[]
for j in [0.,.2,.4,.6,.7,1.]:
 H,st,sp=j1j2(8,j);e,c=ground(H);sign=(-1.)**np.sum(sp[:,::2]<0,axis=1);raw=sign*c
 def bound(v):return float(min(np.sum(v[v>0]**2),np.sum(v[v<0]**2)))
 rows.append(dict(J2=j,E0=float(e[0]),gap=float(e[1]-e[0]),positive_infidelity_bound_Marshall=bound(c),positive_infidelity_bound_original=bound(raw)))
(P/'sign_basis_scan.json').write_text(json.dumps(rows,indent=2))
a=json.loads((P/'sampling_results.json').read_text());b=[]
for method in sorted(set(x['method'] for x in a)):
 q=[x for x in a if x['method']==method];b.append(dict(method=method,repetitions=len(q),symmetry_audit_flags=sum(abs(x['mean_magnetization'])>.1 for x in q),no_sign_changes=sum(x['magnetization_sign_changes']==0 for x in q),mean_naive_SE=float(np.mean([x['naive_magnetization_SE'] for x in q]))))
(P/'symmetry_sampling_audit.json').write_text(json.dumps(b,indent=2))
# Validate the symmetry projection's manual derivatives.
from symmetry import SymRBM
H,st,sp=tfim(6,.2);e,c=ground(H,True,6);net=SymRBM(6,3);p=net.initial(4);v,J,_=net.values(p,sp);E=v@(H@v);g=2*J.T@(H@v-E*v);err=[]
for k in range(len(p)):
 pp=p.copy();pm=p.copy();pp[k]+=1e-6;pm[k]-=1e-6;vp=net.values(pp,sp,False);vm=net.values(pm,sp,False);fd=(vp@(H@vp)-vm@(H@vm))/2e-6;err.append(abs(fd-g[k]))
assert max(err)<1e-7;(P/'symmetry_gradient_check.json').write_text(json.dumps(dict(maximum_error=float(max(err))),indent=2));print(rows,b,max(err))

# Author: Don van Beusichem
from core import *
from scipy.linalg import eigh
# Independent tensor-product construction on small full spaces.
def kronop(ops):
 a=np.array([[1.]])
 for op in ops[::-1]:a=np.kron(a,op)
 return a
X=np.array([[0.,1.],[1.,0.]]);Z=np.diag([-1.,1.]);Y=np.array([[0.,1j],[-1j,0.]]);I=np.eye(2);rows=[]
for n in [4,6]:
 H,st,sp=tfim(n,.7);ref=np.zeros(H.shape)
 for i in range(n):
  ops=[I]*n;ops[i]=X;ref-=.7*kronop(ops);ops=[I]*n;ops[i]=Z;ops[(i+1)%n]=Z;ref-=kronop(ops)
 rows.append(dict(check=f'tensor_TFIM_{n}',maximum_error=float(np.max(abs(H.toarray()-ref)))))
n=6;H,st,sp=j1j2(n,.3,False);ref=np.zeros((1<<n,1<<n),complex)
for dist,J in [(1,1.),(2,.3)]:
 for i in range(n):
  for op in [X,Y,Z]:
   ops=[I]*n;ops[i]=op;ops[(i+dist)%n]=op;ref+=J/4*kronop(ops)
rows.append(dict(check='tensor_J1J2_6',maximum_error=float(np.max(abs(H.toarray()-ref[np.ix_(st,st)])))))
H,st,sp=tfim(6,1.);e,target=ground(H,True,6)
for model in [RBM(6,3),SignedMLP(6,4)]:
 p=model.initial(7);c,J,O=model.values(p,sp);hc=H@c;E=c@hc
 for objective in ['energy','infidelity']:
  overlap=c@target;g=2*J.T@(hc-E*c) if objective=='energy' else -2*overlap*(J.T@target)
  def value(p):
   a=model.values(p,sp,False);return a@(H@a) if objective=='energy' else 1-(a@target)**2
  errors=[]
  for k in range(len(p)):
   pp=p.copy();pm=p.copy();pp[k]+=1e-6;pm[k]-=1e-6;errors.append(abs((value(pp)-value(pm))/2e-6-g[k]))
  rows.append(dict(check=model.name+'_'+objective+'_gradient',maximum_error=float(max(errors))))
 rows.append(dict(check=model.name+'_normalized_tangent',maximum_error=float(np.max(abs(c@J)))))
 if isinstance(model,RBM):
  prob=c*c;oc=O-prob@O;S=oc.T@(prob[:,None]*oc);rows.append(dict(check='QGT_covariance_identity',maximum_error=float(np.max(abs(S-J.T@J)))))
assert max(x['maximum_error'] for x in rows)<1e-7
(P/'implementation_checks.json').write_text(json.dumps(rows,indent=2));print(rows)

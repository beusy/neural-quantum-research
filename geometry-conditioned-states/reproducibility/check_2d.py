# Author: Don van Beusichem
import numpy as np,json
from pathlib import Path
from two_dimensional import Net2D
from train import core
import chemistry as ch
P=Path(__file__).parent;net=Net2D(width=5);p=net.initial(4,core(4,2.5));x=np.array([[1.5,2.1],[3.2,1.7]]);X=(x-2.5)/1.5;H=np.array([ch.getH(4,q,'dimer') for q in x])
def obj(p):
 v,_,ca=net.forward(X,p);n=(v*v).sum(1);Hv=np.einsum('bij,bj->bi',H,v);E=(v*Hv).sum(1)/n;g=2*(Hv-E[:,None]*v)/n[:,None]/len(x);return E.mean(),net.backward(g,ca)
f,g=obj(p);err=[]
for k in np.random.default_rng(1).choice(len(p),40,replace=False):
 pp=p.copy();pm=p.copy();pp[k]+=1e-6;pm[k]-=1e-6;err.append(abs((obj(pp)[0]-obj(pm)[0])/2e-6-g[k]))
model=np.load(P/'dimer_n25_s0.npz');net=Net2D();p=model['p'];x=np.array([[2.2,2.8]]);v,dv,_=net.forward((x-2.5)/1.5,p);h=1e-5;H=ch.getH(4,x[0],'dimer');norm=(v*v).sum();E=(v@H@v.T).item()/norm;res=H@v[0]-E*v[0];forceerr=[]
for k in range(2):
 xp=x.copy();xm=x.copy();xp[0,k]+=h;xm[0,k]-=h;hp=(ch.getH(4,xp[0],'dimer')-ch.getH(4,xm[0],'dimer'))/(2*h);force=-(v[0]@hp@v[0]+2*dv[0,k]@res)/norm
 def energy(z):
  q=net.forward((z-2.5)/1.5,p)[0][0];return q@ch.getH(4,z[0],'dimer')@q/(q@q)
 fd=-(energy(xp)-energy(xm))/(2*h);forceerr.append(abs(force-fd))
out={'two_dimensional_parameter_gradient_max':float(max(err)),'two_dimensional_force_check_max':float(max(forceerr))};assert max(err)<1e-7 and max(forceerr)<1e-6;(P/'two_dimensional_checks.json').write_text(json.dumps(out,indent=2));print(out)

# Author: Don van Beusichem
from pathlib import Path
import numpy as np,json
from neural import CoefficientNet,fit,quantities
from train import core
import chemistry as c
rr=np.random.default_rng(23);net=CoefficientNet(4,5);r=np.array([1.3,2.,3.5]);H=np.array([c.getH(2,x) for x in r]);base=core(2,3.)
# Run zero iterations only to obtain an objective closure; check independently chosen parameters.
opt,obj=fit(net,r,H,0,base,penalty=1.,maxiter=0);p=net.initial(8,base);value,g=obj(p);errs=[]
for k in range(len(p)):
 h=1e-6;pp=p.copy();pm=p.copy();pp[k]+=h;pm[k]-=h;fd=(obj(pp)[0]-obj(pm)[0])/(2*h);errs.append(abs(fd-g[k]))
out={'mixed_parameter_gradient_max':float(max(errs))};print(out);assert max(errs)<1e-7
open(Path(__file__).parent/'gradient_checks.json','w').write(json.dumps(out,indent=2))

# Author: Don van Beusichem
"""Fetch the exact rMD17 ethanol bytes used in this report, verifying integrity."""
from pathlib import Path
import urllib.request,hashlib
p=Path(__file__).parent/'data';p.mkdir(exist_ok=True)
f=p/'rmd17_ethanol.npz';expected='0824d78c687b8ede25f7162c54c5c6434cfe37f9f859ee8f223c60379d7d5cf9'
if not f.exists(): urllib.request.urlretrieve('https://ndownloader.figshare.com/files/62265733',f)
actual=hashlib.sha256(f.read_bytes()).hexdigest()
if actual!=expected: raise RuntimeError(f'Dataset hash mismatch: {actual}. Do not overwrite the recorded reference silently.')
print('Verified',f)

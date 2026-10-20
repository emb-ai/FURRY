from pathlib import Path
import gzip,hashlib,json,os,shutil,time
import h5py,numpy as np
p=Path('/home/jovyan/vlad/FURRY/cup-homie-finetune-20261015')
expected='05376522259b4eb9091a80d41a31b0a768b3933349a231d69de5580fdb32b9cd'
assert (p/'new-cup.hdf5.gz').stat().st_size==670408989
assert not (p/'new-cup.hdf5').exists()
with gzip.open(p/'new-cup.hdf5.gz','rb') as src,(p/'new-cup.hdf5.partial').open('wb') as dst:shutil.copyfileobj(src,dst)
h=hashlib.sha256()
with (p/'new-cup.hdf5.partial').open('rb') as f:
 for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
assert h.hexdigest()==expected
with h5py.File(p/'new-cup.hdf5.partial','r') as f:
 names=[x.decode() for x in f['mask/train'][:]]
 assert len(names)==len(set(names))==59 and len(f['mask/valid'])==0
 assert f.attrs['schema']=='furry-visual-il-v1'
 frames=0
 for name in names:
  d=f['data/'+name];a=d['actions'][:];assert a.shape[1]==39 and np.isfinite(a).all()
  c=d['obs/previous_command'][:];assert np.isfinite(c).all() and np.array_equal(c[1:],a[:-1])
  assert d['obs/agentview_image'].shape==(len(a),224,224,3)
  frames+=len(a)
 assert frames==16042
(p/'new-cup.hdf5.partial').replace(p/'new-cup.hdf5')
r={'verified':True,'sha256':expected,'episodes':59,'frames':frames,'all_train':True,'verified_at':time.time()}
(p/'upload-verification.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)

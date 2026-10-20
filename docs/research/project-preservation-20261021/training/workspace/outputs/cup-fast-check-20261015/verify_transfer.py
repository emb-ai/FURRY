import gzip
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
import h5py

root=Path(__file__).resolve().parent
manifest=json.loads((root/'transfer-manifest.json').read_text())
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
for spec in manifest['files']:
    path=root/spec['remote_name']
    assert path.stat().st_size==spec['bytes'], path
    assert digest(path)==spec['sha256'], path
target=root/'cup44-no-shadow-causal.hdf5'
if not target.exists():
    partial=root/'cup44-no-shadow-causal.hdf5.partial'
    with gzip.open(root/'cup44-no-shadow-causal.hdf5.gz','rb') as src, partial.open('wb') as dst:
        shutil.copyfileobj(src,dst,8*1024*1024)
    assert digest(partial)==manifest['uncompressed_hdf5_sha256']
    partial.replace(target)
assert digest(target)==manifest['uncompressed_hdf5_sha256']
with h5py.File(target,'r') as h:
    train=set(h['mask/train'][:]); valid=set(h['mask/valid'][:])
    assert len(train)==40 and len(valid)==4 and not train&valid
    assert len(h['data'])==44
with tarfile.open(root/'dp-code.tar.gz','r:gz') as archive:
    for member in archive.getmembers():
        assert (root/member.name).resolve().is_relative_to(root)
        assert not member.issym() and not member.islnk()
    archive.extractall(root)
result={'verified':True,'sha256':manifest['uncompressed_hdf5_sha256'],'train_episodes':40,'validation_episodes':4,'all_uploaded_files_hash_match':True}
(root/'transfer-verified.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)

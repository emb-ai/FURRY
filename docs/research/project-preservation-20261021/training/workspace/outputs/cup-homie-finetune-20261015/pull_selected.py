from pathlib import Path
import json,subprocess,tarfile,hashlib,time,re
O=Path(__file__).resolve().parent
names=json.loads((O/'selected-source-episodes.json').read_text())
assert all(re.fullmatch(r'episode-\d+',n) for n in names)
adb=[str(O.parents[1]/'.tools/platform-tools/adb'),'-s','2G0YC5ZG7T05SR','exec-out','run-as','org.g1lab.quest']
dest=O/'recordings';dest.mkdir(exist_ok=True)
archive=O/'device-recordings.tar'
with archive.open('wb') as f:
    subprocess.run(adb+['tar','-C','files/g1/recordings','-cf','-',*names],stdout=f,check=True)
with tarfile.open(archive) as t:
    for m in t:
        if not (m.isfile() or m.isdir()) or not (dest/m.name).resolve().is_relative_to(dest.resolve()):raise ValueError(m.name)
    t.extractall(dest,filter='data')
print('Copied',len(names),'episodes',archive.stat().st_size,'bytes',flush=True)
cmd='cd files/g1/recordings && sha256sum '+' '.join(n+'/*' for n in names)
lines=subprocess.check_output(adb+['sh','-c',cmd],text=True).splitlines();verified=[]
for line in lines:
    sha,name=line.split(maxsplit=1);p=dest/name;h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    assert h.hexdigest()==sha,name
    verified.append({'path':name,'sha256':sha,'bytes':p.stat().st_size})
(O/'pull-verification.json').write_text(json.dumps({'serial':'2G0YC5ZG7T05SR','episodes':names,'files':verified,'checked_at':time.time(),'device_modified':False},indent=2))
print('Verified',len(verified),'files',flush=True)

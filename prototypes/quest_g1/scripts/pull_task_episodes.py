"""Copy finalized visual-learning episodes from Quest, with device SHA256 checks."""
import argparse
import hashlib
import io
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
import shutil
import json


def pull(destination, serial=None):
    root=Path(__file__).resolve().parents[1]
    adb=str(root/'.tools/platform-tools/adb')
    if serial is None:
        rows=subprocess.check_output([adb,'devices'],text=True).splitlines()[1:]
        devices=[r.split()[0] for r in rows if len(r.split())>=2 and r.split()[1]=='device']
        if len(devices)!=1: raise RuntimeError('Connect one Quest via USB or pass --serial')
        serial=devices[0]
    prefix=[adb,'-s',serial,'exec-out','run-as','org.g1lab.quest']
    def read(*args):return subprocess.check_output(prefix+list(args))
    entries=read('find','files/g1/recordings','-name','learning-result.json').decode().splitlines()
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True);saved=[]
    for entry in entries:
        name=Path(entry).parent.name
        if not re.fullmatch(r'episode-\d+',name): raise ValueError('Unexpected episode name')
        remote='files/g1/recordings/'+name
        complete=subprocess.run(prefix+['test','-f',remote+'/complete.json'],capture_output=True)
        if complete.returncode:continue
        hashes={}
        for row in read('sh','-c',f'cd {remote} && sha256sum *').decode().splitlines():
            sha,filename=row.split(maxsplit=1);filename=filename.lstrip('*')
            if '/' in filename or filename.startswith('.'):raise ValueError('Unexpected capture filename')
            hashes[filename]=sha
        target=destination/name
        if not target.exists():
            with tempfile.TemporaryDirectory(dir=destination,prefix='.pull-') as temp:
                archive=Path(temp)/'episode.tar'
                with archive.open('wb') as f:
                    subprocess.run(prefix+['tar','-C','files/g1/recordings','-cf','-',name],stdout=f,check=True)
                with tarfile.open(archive) as tar:
                    for member in tar.getmembers():
                        if not (member.isfile() or member.isdir()) or not (Path(temp)/member.name).resolve().is_relative_to(Path(temp).resolve()):
                            raise ValueError('Unexpected archive member')
                    tar.extractall(temp,filter='data')
                copied=Path(temp)/name
                verify(copied,hashes)
                shutil.move(str(copied),target)
        verify(target,hashes)
        saved.append({'episode':name,'files':len(hashes),'sha256':hashes})
        print(f'Verified {name}: {len(hashes)} files',flush=True)
    receipt={'serial':serial,'episodes':saved,'device_files_deleted':False}
    (destination/'pull-verification.json').write_text(json.dumps(receipt,indent=2))
    if not saved:raise RuntimeError('No finalized visual-learning episodes yet. Finish a task recording with Y.')
    return receipt


def verify(folder, hashes):
    for name,expected in hashes.items():
        h=hashlib.sha256()
        with (folder/name).open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        if h.hexdigest()!=expected:raise ValueError(f'Checksum mismatch: {folder/name}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path);p.add_argument('--serial')
    a=p.parse_args();pull(a.destination,a.serial)

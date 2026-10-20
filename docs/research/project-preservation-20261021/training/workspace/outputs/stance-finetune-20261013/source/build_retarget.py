from pathlib import Path
import subprocess, json, hashlib, shutil
import mujoco, onnxruntime
R=Path.cwd(); O=R/'outputs/stance-finetune-20261013'; N=O/'source/native'; N.mkdir(exist_ok=True)
repo=R/'.tools/furry-publish'; rev='3d0164f24ad8c490b158039612bda376084b4767'; prefix='prototypes/quest_g1/android/native/'
files=subprocess.check_output(['git','-C',str(repo),'ls-tree','-r','--name-only',rev,prefix],text=True).splitlines()
for f in files:
 if Path(f).suffix in ('.h','.cpp'):(N/Path(f).name).write_bytes(subprocess.check_output(['git','-C',str(repo),'show',rev+':'+f]))
provenance=json.loads((R/'outputs/quest-finetune-20261009/author-mixture/video-baseline-main/provenance.json').read_text())
for name,digest in provenance['source_sha256'].items():assert hashlib.sha256((N/name).read_bytes()).hexdigest()==digest,name
s=(N/'meta_retarget.cpp').read_text(); needle='    gmr.SetTargets(targets);'; assert s.count(needle)==1
ref=(R/'outputs/stance-width-audit-20261013/meta_retarget.cpp').read_text(); block=ref[ref.index('    // Diagnostic intervention:'):ref.index(needle)]
s=s.replace(needle,block+needle);s='#include <cstdlib>\n'+s;(N/'meta_retarget.cpp').write_text(s)
shutil.copy2(R/'outputs/quest-finetune-20261009/author-mixture/video-baseline-main/src/retarget.cpp',O/'source/retarget.cpp')
mj=Path(mujoco.__file__).parent;ort=Path(onnxruntime.__file__).parent/'capi'; headers=R/'outputs/full-body-apk-20261010/source/vendor/onnxruntime-android/headers'
cmd=['c++','-O2','-std=c++17','-I'+str(N),'-I'+str(mj/'include'),'-I'+str(headers),str(O/'source/retarget.cpp'),str(N/'gmr.cpp'),str(N/'meta_retarget.cpp'),str(N/'simulation.cpp'),str(next(mj.glob('libmujoco*.dylib'))),str(next(ort.glob('libonnxruntime*.dylib'))),'-Wl,-rpath,'+str(mj),'-Wl,-rpath,'+str(ort),'-o',str(O/'retarget')]
(O/'build-command.json').write_text(json.dumps(cmd,indent=2));subprocess.run(cmd,check=True)
(O/'retarget-provenance.json').write_text(json.dumps({'base_revision':rev,'original_verified':provenance['source_sha256'],'intervention':block,'source_hashes':{str(p.relative_to(O)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (O/'source').rglob('*') if p.is_file()}},indent=2))

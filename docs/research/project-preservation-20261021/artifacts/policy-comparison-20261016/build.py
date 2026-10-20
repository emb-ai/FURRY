from pathlib import Path
import subprocess,mujoco,onnxruntime,json,hashlib
R=Path.cwd();O=R/'outputs/policy-comparison-20261016';P=R/'outputs/training-capture-20261015/project';N=P/'android/native'
m=Path(mujoco.__file__).parent;r=Path(onnxruntime.__file__).parent/'capi'
c=['c++','-O2','-std=c++17','-I'+str(N),'-I'+str(m/'include'),'-I'+str(P/'vendor/onnxruntime-android/headers'),str(O/'native.cpp'),*[str(N/(s+'.cpp')) for s in ['simulation','gmr','meta_retarget']],str(next(m.glob('libmujoco*.dylib'))),str(next(r.glob('libonnxruntime*.dylib'))),'-Wl,-rpath,'+str(m),'-Wl,-rpath,'+str(r),'-o',str(O/'native')]
subprocess.run(c,check=True)
(O/'native-source.json').write_text(json.dumps({str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in N.glob('*') if p.suffix in ['.cpp','.h']},indent=2))

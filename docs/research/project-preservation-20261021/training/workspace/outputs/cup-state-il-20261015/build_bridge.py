from pathlib import Path
import subprocess,sys
import mujoco,onnxruntime
O=Path(__file__).resolve().parent;mj=Path(mujoco.__file__).parent;ort=Path(onnxruntime.__file__).parent/'capi';mac=sys.platform=='darwin'
mlib=next(mj.glob('libmujoco*.dylib' if mac else 'libmujoco.so.*'));olib=next(ort.glob('libonnxruntime*.dylib' if mac else 'libonnxruntime.so.*'))
runtime=O/'runtime-libs';runtime.mkdir(exist_ok=True)
if not mac:
    link=runtime/'libonnxruntime.so.1'
    if not link.exists():link.symlink_to(olib)
cmd=['c++','-O2','-std=c++17','-dynamiclib' if mac else '-shared','-fPIC','-I'+str(O/'native'),'-I'+str(mj/'include'),'-I'+str(O/'ort-headers'),str(O/'bridge_fresh.cpp'),str(O/'native/simulation.cpp'),str(mlib),str(olib),'-Wl,-rpath,'+str(ort),'-Wl,-rpath,'+str(mj),'-o',str(O/('bridge.dylib' if mac else 'bridge.so'))]
cmd.extend(['-Wl,-rpath,'+str(runtime)])
subprocess.run(cmd,check=True)

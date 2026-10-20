from pathlib import Path
import subprocess,mujoco,onnxruntime
O=Path(__file__).resolve().parent
P=O.parent/'rubik-table-rest-20261015/project'
mj=Path(mujoco.__file__).parent;ort=Path(onnxruntime.__file__).parent/'capi'
subprocess.run(['c++','-O2','-std=c++17','-dynamiclib','-I'+str(O/'native'),'-I'+str(mj/'include'),'-I'+str(P/'vendor/onnxruntime-android/headers'),str(O/'bridge.cpp'),str(O/'native/simulation.cpp'),str(next(mj.glob('libmujoco*.dylib'))),str(next(ort.glob('libonnxruntime*.dylib'))),'-Wl,-rpath,'+str(ort),'-Wl,-rpath,'+str(mj),'-o',str(O/'bridge.dylib')],check=True)

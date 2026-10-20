from pathlib import Path
import subprocess,mujoco,onnxruntime
O=Path(__file__).resolve().parent;mj=Path(mujoco.__file__).parent;ort=Path(onnxruntime.__file__).parent/'capi'
subprocess.run(['c++','-O2','-std=c++17','-shared','-fPIC','-I'+str(O/'native'),'-I'+str(mj/'include'),'-I'+str(O/'ort-headers'),str(O/'bridge_fresh.cpp'),str(O/'native/simulation.cpp'),str(next(mj.glob('libmujoco.so*'))),str(next(ort.glob('libonnxruntime.so*'))),'-Wl,-rpath,'+str(ort),'-Wl,-rpath,'+str(mj),'-o',str(O/'bridge.so')],check=True)

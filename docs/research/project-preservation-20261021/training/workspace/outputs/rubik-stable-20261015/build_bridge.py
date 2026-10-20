from pathlib import Path
import sys,os,subprocess
p=Path(__file__).resolve().parent;sys.path.insert(0,str(p.parent/'rubik-canonical-20261015'));os.environ['BRIDGE']='bridge-original-snapshot.dylib'
from common import *
native=p.parent/'rubik-canonical-20261015/original-native'
subprocess.run(['c++','-O2','-std=c++17','-dynamiclib','-I'+str(native),'-I'+str(mj/'include'),'-I'+str(P/'vendor/onnxruntime-android/headers'),str(p/'contact_bridge.cpp'),str(native/'simulation.cpp'),str(next(mj.glob('libmujoco*.dylib'))),str(next(ort.glob('libonnxruntime*.dylib'))),'-Wl,-rpath,'+str(ort),'-o',str(p.parent/'rubik-sequences-20261015/bridge-contacts.dylib')],check=True)

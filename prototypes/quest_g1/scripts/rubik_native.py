"""Desktop binding to the exact Quest puzzle mechanism for physical probes."""
import ctypes as C
from pathlib import Path
import os
import subprocess
import sys
import mujoco

ROOT=Path(__file__).resolve().parents[1]


class NativePuzzle:
    def __init__(self,model,data,build_folder):
        self.model,self.data=model,data
        folder=Path(build_folder);folder.mkdir(parents=True,exist_ok=True)
        mj=Path(mujoco.__file__).parent
        lib=next(mj.glob('libmujoco*.dylib' if sys.platform=='darwin' else 'libmujoco.so*'))
        target=folder/('rubik-probe.dylib' if sys.platform=='darwin' else 'rubik-probe.so')
        subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O2','-shared','-fPIC',
            '-I'+str(ROOT/'android/native'),'-I'+str(mj/'include'),
            str(ROOT/'scripts/rubik_probe_bridge.cpp'),str(lib),'-Wl,-rpath,'+str(mj),'-o',str(target)],check=True)
        self.lib=C.CDLL(str(target));p=C.c_void_p
        for name,args,result in [('create',[p],p),('reset',[p,p,p],None),('begin',[p,p,p,C.c_double,C.c_double],None),
                                ('finish',[p,p,p],None),('activate',[p,p,p,C.c_int],C.c_int),('solved',[p,p],C.c_int),
                                ('contacts',[p,p,p,C.c_int],C.c_int),('destroy',[p],None)]:
            f=getattr(self.lib,'rubik_'+name);f.argtypes=args;f.restype=result
        self.handle=self.lib.rubik_create(model._address)
    def reset(self):self.lib.rubik_reset(self.handle,self.model._address,self.data._address)
    def begin(self,left=0,right=0):self.lib.rubik_begin(self.handle,self.model._address,self.data._address,left,right)
    def finish(self):self.lib.rubik_finish(self.handle,self.model._address,self.data._address)
    def activate(self,face):return self.lib.rubik_activate(self.handle,self.model._address,self.data._address,face)
    def solved(self):return bool(self.lib.rubik_solved(self.handle,self.data._address))
    def contacts(self,hand):return self.lib.rubik_contacts(self.handle,self.model._address,self.data._address,hand)
    def close(self):
        if self.handle:self.lib.rubik_destroy(self.handle);self.handle=None

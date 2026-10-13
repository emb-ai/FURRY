"""Use the exact native clearance function for desktop render QA."""
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

class CameraFade:
    def __init__(self):
        self.folder=tempfile.TemporaryDirectory(prefix='furry-camera-fade-')
        library=Path(self.folder.name)/('fade.dylib' if sys.platform=='darwin' else 'fade.so')
        subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O2','-shared','-fPIC',
                        '-I'+str(ROOT/'android/native'),str(ROOT/'scripts/camera_fade_bridge.cpp'),'-o',str(library)],check=True)
        self.library=ctypes.CDLL(str(library));self.function=self.library.camera_alpha
        pointer=ctypes.POINTER(ctypes.c_float)
        self.function.argtypes=[pointer,pointer,ctypes.c_int,pointer];self.function.restype=ctypes.c_float
    def alpha(self,model,vertices,eye):
        values=[np.ascontiguousarray(np.asarray(model,dtype=np.float32).T),
                np.ascontiguousarray(vertices,dtype=np.float32),np.ascontiguousarray(eye,dtype=np.float32)]
        pointer=ctypes.POINTER(ctypes.c_float)
        return self.function(values[0].ctypes.data_as(pointer),values[1].ctypes.data_as(pointer),len(vertices),values[2].ctypes.data_as(pointer))

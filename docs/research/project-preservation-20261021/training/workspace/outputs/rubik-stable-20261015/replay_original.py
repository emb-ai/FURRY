from pathlib import Path
import os,sys
p=Path(__file__).resolve().parent
sys.path.insert(0,str(p.parent/'rubik-canonical-20261015'))
os.environ['BRIDGE']='bridge-original-home.dylib'
from common import *
O=p
exec((p.parent/'rubik-canonical-20261015/replay.py').read_text().replace('from common import *','').replace("z=np.load(O/(name+'.npz'))","z=dict(np.load(O/(name+'.npz')))"))

"""Native diagnostic session. Original bridge exposes no individual finger API."""
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent;PREV=HERE.parent/'rubik-regrasp-20261015';DATA=HERE.parent/'rubik-stable-20261015'
os.environ.update(BRIDGE='bridge-original-snapshot.dylib',ROLL='-45',REBASE='1',PIVOT_X='0')
__file__=str(PREV/'turn_series.py');exec((PREV/'turn_series.py').read_text().split('try:\n for cycle')[0]);O=HERE
assert not hasattr(lib,'fingers')
lib.snapshot.argtypes=[ptr];lib.snapshot.restype=ptr;lib.restore.argtypes=[ptr,ptr];lib.free_snapshot.argtypes=[ptr]
arrays=[trace,qs,vs,states,support,commands,actions]
def capture():return (lib.snapshot(s.p),elapsed,phase,goals.copy(),grips.copy(),len(trace),len(events))
def restore_state(st):
 global elapsed,phase,goals,grips
 lib.restore(s.p,st[0]);s.read();elapsed,phase=st[1:3];goals=st[3].copy();grips=st[4].copy()
 for a in arrays:del a[st[5]:]
 del events[st[6]:]
z=dict(np.load(DATA/'five-cycles.npz'));anchors={}
for k in range(len(qs),len(z['qpos'])):
 phase=int(z['trace'][k,1]);cc=z['commands'][k];tick(cc[:14].reshape(2,7),cc[-2:])
 if k==2089:
  homeC,homeR=frame();homeG=goals.copy();homeP=s.poses();homeF=fingers();homeLocal=homeP.copy()
  for h in range(2):homeLocal[h,:3]=homeR.inv().apply(homeP[h,:3]-homeC);homeLocal[h,3:]=(homeR.inv()*R.from_quat(homeP[h,[4,5,6,3]])).as_quat()[[3,0,1,2]]
 if k in [11700,13089,13958]:anchors[{11700:'four',13089:'grasp',13958:'five'}[k]]=capture();print('ANCHOR',k,flush=True)
assert np.max(abs(q-z['qpos'][-1]))==0
print('SESSION READY',flush=True)
try:
 for line in sys.stdin:
  request=json.loads(line)
  if request.get('quit'):break
  try:
   restore_state(anchors[request.get('anchor','five')]);exec(Path(request['run_file']).read_text(),globals());print('SESSION DONE',flush=True)
  except Exception:
   import traceback;traceback.print_exc();print('SESSION ERROR',flush=True)
finally:
 for st in anchors.values():lib.free_snapshot(st[0])
 s.close()

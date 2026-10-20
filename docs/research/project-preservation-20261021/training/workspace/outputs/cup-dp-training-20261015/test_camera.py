from pathlib import Path
import numpy as np,json
from capture_snapshot import inspect_episode
from training_renderer import TrainingRenderer
O=Path(__file__).resolve().parent
e=inspect_episode(O.parent/'task-demo-audit-20261015/recordings/episode-1792017537461911');r=TrainingRenderer(e);r.model.light_castshadow[:]=0
try:
 idx=int(e['indices'][100]);prev=int(e['indices'][99]);a=r.render_observation(idx,prev)
 e['camera']['gaze_yaw'][idx]+=1.;e['camera']['gaze_pitch'][idx]+=.5
 e['camera']['camera_qw'][idx]=1.;e['camera']['camera_qx'][idx]=e['camera']['camera_qy'][idx]=e['camera']['camera_qz'][idx]=0.
 b=r.render_observation(idx,prev);assert np.array_equal(a,b),'Current predicted gaze leaks into image'
 e['camera']['gaze_yaw'][prev]+=.3
 c=r.render_observation(idx,prev);assert not np.array_equal(a,c),'Previous gaze not applied'
 result={'current_gaze_changes_observation':False,'previous_gaze_changes_observation':True,'shadows':False};(O/'camera-test.json').write_text(json.dumps(result,indent=2));print(result)
finally:r.close()

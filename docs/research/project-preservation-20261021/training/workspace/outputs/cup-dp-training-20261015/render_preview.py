from pathlib import Path
import subprocess
from PIL import Image
from capture_snapshot import inspect_episode
from training_renderer import TrainingRenderer
O=Path(__file__).resolve().parent;e=inspect_episode(O.parent/'task-demo-audit-20261015/recordings/episode-1792017537461911');r=TrainingRenderer(e);r.model.light_castshadow[:]=0
p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','640x360','-r','30','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(O/'cup-no-shadow.mp4')],stdin=subprocess.PIPE)
try:
 for k,idx in enumerate(e['indices']):
  r.render_observation(idx,e['indices'][max(0,k-1)]);rgb=r.renderer.render();p.stdin.write(rgb.tobytes())
  if k==len(e['indices'])//2:Image.fromarray(rgb).save(O/'cup-no-shadow.jpg')
finally:r.close();p.stdin.close()
assert p.wait()==0

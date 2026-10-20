from pathlib import Path
import sys, subprocess, json
from PIL import Image
O=Path(__file__).resolve().parent
sys.path.insert(0,str(O.parent/'rubik-table-rest-20261015/project'))
from g1_sim.visual_dataset import inspect_episode, EpisodeRenderer
name='episode-1792017537461911'
e=inspect_episode(O.parent/'task-demo-audit-20261015/recordings'/name)
out=O/'cup-trajectory-46cm.mp4'
p=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','640x360','-r','30','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
r=EpisodeRenderer(e)
try:
 for k,idx in enumerate(e['indices']):
  r.render(idx)
  rgb=r.renderer.render()
  p.stdin.write(rgb.tobytes())
  if k==len(e['indices'])//2: Image.fromarray(rgb).save(O/'cup-trajectory-preview.jpg')
finally:
 r.close();p.stdin.close()
if p.wait()!=0: raise RuntimeError('ffmpeg failed')
print(json.dumps({'video':str(out),'episode':name,'frames':len(e['indices']),'fps':30}))

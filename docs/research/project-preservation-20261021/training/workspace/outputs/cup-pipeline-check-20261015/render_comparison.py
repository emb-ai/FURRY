from pathlib import Path
import sys,subprocess
import numpy as np,mujoco
from PIL import Image,ImageDraw
O=Path(__file__).resolve().parent;episode='episode-1792014836827567'
raw=O.parent/'task-demo-audit-20261015/recordings'/episode
m=mujoco.MjModel.from_binary_path(str(raw/'learning-model.mjb'));m.light_castshadow[:]=0
d=mujoco.MjData(m);r=mujoco.Renderer(m,height=420,width=640)
camera=mujoco.MjvCamera();camera.lookat[:]=[.3,0,.95];camera.distance=1.65;camera.azimuth=130;camera.elevation=-25
clips=[np.load(O/f'{label}-policy-h48-{episode}-s0.npz') for label in ('baseline','latest')]
out=O/'early-vs-final-train.mp4'
p=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x460','-r','20','-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(out)],stdin=subprocess.PIPE)
try:
 for frame in range(220):
  t=frame/20;canvas=Image.new('RGB',(1280,460),'#101c25');draw=ImageDraw.Draw(canvas)
  for side,z in enumerate(clips):
   j=min(len(z['qpos'])-1,max(0,np.searchsorted(z['trace'][:,0],t,side='right')-1))
   d.qpos[:]=z['qpos'][j];d.qvel[:]=z['qvel'][j];d.mocap_pos.flat[:]=z['mocap_pos'];d.mocap_quat.flat[:]=z['mocap_quat'];mujoco.mj_forward(m,d)
   r.update_scene(d,camera=camera);r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
   canvas.paste(Image.fromarray(r.render()),(640*side,40))
   draw.text((640*side+16,12),('EPOCH 15: FAIL' if side==0 else 'EPOCH 300: SUCCESS at 8.7s')+f' | {t:.1f}s',fill='white')
  if frame==50:canvas.save(O/'early-vs-final-preview.jpg')
  p.stdin.write(np.asarray(canvas).tobytes())
finally:r.close();p.stdin.close()
assert p.wait()==0

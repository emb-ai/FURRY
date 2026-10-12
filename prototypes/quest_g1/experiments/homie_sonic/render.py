from pathlib import Path
import sys,json,subprocess
import numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
APP=Path(__file__).resolve().parents[2];out=Path(sys.argv[1]);clip=sys.argv[2] if len(sys.argv)>2 else 'episode-1791652646384286'
model=mujoco.MjModel.from_xml_path(str(APP/'android/assets/scene-stand.xml'));data=mujoco.MjData(model)
backends=['twist2','homie','sonic'];labels=['TWIST2 +500','HOMIE v2','SONIC VR3 + planner'];colors=['#cccccc','#38ced8','#ffa051']
traces=[np.load(out/'final/live'/f'{b}-{clip}-s11.npz') for b in backends];rows=[json.loads((out/'final/live'/f'{b}-{clip}-s11.rows.json').read_text()) for b in backends]
count=min(len(t['state']) for t in traces);fps=20;frames=int(count/50*fps);size=(1440,610)
renderer=mujoco.Renderer(model,height=400,width=480);cam=mujoco.MjvCamera();cam.distance=3.2;cam.elevation=-22;cam.azimuth=125
font=lambda n:ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',n)
video=out/'live-comparison.mp4';proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{size[0]}x{size[1]}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
try:
 for frame in range(frames):
  i=min(round(frame/fps*50),count-1);canvas=Image.new('RGB',size,'#131b23');draw=ImageDraw.Draw(canvas)
  draw.text((20,12),f'FURRY | recorded Meta input → fresh MuJoCo rollouts | t={frame/fps:.2f}s',font=font(23),fill='white')
  draw.text((20,44),'Same camera goal and upper-body target. Autonomous modes do not copy recorded leg poses.',font=font(18),fill='#b2bdc8')
  for k,b in enumerate(backends):
   state=traces[k]['state'][i];data.qpos[:]=state[:model.nq];mujoco.mj_forward(model,data)
   # Same world-centred camera in all panels; trajectory lag remains visible.
   ref=traces[k]['reference'][i];cam.lookat[:]=[ref[0],ref[1],.65]
   renderer.update_scene(data,camera=cam);canvas.paste(Image.fromarray(renderer.render()),(k*480,105));draw=ImageDraw.Draw(canvas)
   r=rows[k][i];draw.text((k*480+16,78),labels[k],font=font(23),fill=colors[k])
   draw.text((k*480+16,516),f'XY {r["root_xy_m"]*100:.1f} cm | contact slip {r["contact_slip_m_s"]*100:.1f} cm/s',font=font(18),fill=colors[k])
   draw.text((k*480+16,546),f'Wrist local {r["wrist_local_mean_m"]*100:.1f} cm',font=font(18),fill='white')
   if r['upward_contact_events']:draw.text((k*480+16,572),'UPWARD CONTACT CANDIDATE',font=font(16),fill='#ff6677')
  if frame==100:canvas.save(out/'video-preview.png')
  proc.stdin.write(canvas.tobytes())
  if frame%100==0:print(frame,frames,flush=True)
 proc.stdin.close();assert proc.wait()==0
finally:renderer.close()

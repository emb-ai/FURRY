from pathlib import Path
import sys,json
import numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
APP=Path(__file__).resolve().parents[2];out=Path(sys.argv[1]);events=[]
for group in ['validation','live']:
 for p in sorted((out/'final'/group).glob('twist2-*.rows.json')):
  rows=json.loads(p.read_text())
  for i,r in enumerate(rows):
   if r['upward_contact_events']:events.append((p,i,r['time']))
m=mujoco.MjModel.from_xml_path(str(APP/'android/assets/scene-stand.xml'));d=mujoco.MjData(m);cam=mujoco.MjvCamera();cam.distance=1.2;cam.elevation=-16;cam.azimuth=100
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',14);sheet=Image.new('RGB',(1200,220*len(events)),'#18222b');draw=ImageDraw.Draw(sheet)
renderer=mujoco.Renderer(m,width=240,height=185)
try:
 for row,(p,idx,t) in enumerate(events):
  trace=np.load(p.with_name(p.name.replace('.rows.json','.npz')));states=trace['state']
  draw.text((8,row*220+5),p.name.replace('twist2-','').replace('.rows.json','')+f' | candidate t={t:.2f}s | -60,-20,0,+40,+100 ms',font=font,fill='white')
  for col,delta in enumerate([-3,-1,0,2,5]):
   i=min(max(0,idx+delta),len(states)-1);sv=states[i];d.qpos[:]=sv[:m.nq];mujoco.mj_forward(m,d)
   feet=np.array([d.body(s+'_ankle_roll_link').xpos for s in ['left','right']]);cam.lookat[:]=feet.mean(0);cam.lookat[2]=.18
   renderer.update_scene(d,camera=cam);sheet.paste(Image.fromarray(renderer.render()),(col*240,row*220+30))
finally:renderer.close()
sheet.save(out/'contact-candidates.png');(out/'contact-candidates.json').write_text(json.dumps([dict(clip=p.name,time=t) for p,i,t in events],indent=2)+'\n')

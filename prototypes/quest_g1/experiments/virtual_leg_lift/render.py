from pathlib import Path
import sys,subprocess,json
import numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
APP=Path(__file__).resolve().parents[2];out=Path(sys.argv[1]);sources=[np.load(out/f'{v}.npz') for v in ['off','on']];rolls=[np.load(out/'rollouts'/f'twist2-{v}-s11.npz') for v in ['off','on']];rows=[json.loads((out/f'common-{v}.rows.json').read_text()) for v in ['off','on']]
a=sources[1]['source'];raw=a[:,86:184].reshape(-1,14,7);patched=a[:,184:282].reshape(-1,14,7)
m=mujoco.MjModel.from_xml_path(str(APP/'android/assets/scene-stand.xml'));d=mujoco.MjData(m);qa=rolls[0]['qa'];renderer=mujoco.Renderer(m,height=280,width=540)
hm=mujoco.MjModel.from_xml_string('<mujoco><visual><global offwidth="540" offheight="280"/></visual><worldbody><light pos="0 0 4"/><geom type="plane" size="100 100 .1" rgba=".2 .23 .27 1"/></worldbody></mujoco>');hd=mujoco.MjData(hm);mujoco.mj_forward(hm,hd);hr=mujoco.Renderer(hm,height=280,width=540)
cam=mujoco.MjvCamera();cam.distance=2.75;cam.elevation=-16;cam.azimuth=130
hc=mujoco.MjvCamera();hc.distance=2.5;hc.elevation=-14;hc.azimuth=130
edges=[(0,1),(0,2),(2,4),(4,6),(0,3),(3,5),(5,7),(1,8),(8,10),(10,12),(1,9),(9,11),(11,13)]
font=lambda n:ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',n)
font20,font16,font14=font(20),font(16),font(14)
size=(1080,1050);fps=20;count=min(len(x['state']) for x in rolls);total=int(count/50*fps)
video=out/'comparison-3-levels.mp4';proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{size[0]}x{size[1]}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
def skeleton(q,reference):
 p=q[:,:3][:,[0,2,1]].copy();p[:,1]*=-1;center=reference[0,:3][[0,2,1]].copy();center[1]*=-1
 hc.lookat[:]=[center[0],center[1],.72];hr.update_scene(hd,camera=hc)
 for i,j in edges:
  g=hr.scene.geoms[hr.scene.ngeom];color=[.9,.65,.20,1] if i in [2,3,4,5] or j in [2,3,4,5,6,7] else [.3,.8,.85,1]
  mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color,np.float32));mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.017,p[i],p[j]);hr.scene.ngeom+=1
 for xyz in p:
  g=hr.scene.geoms[hr.scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.026]*3),xyz,np.eye(3).ravel(),np.array([.9,.9,.9,1],np.float32));hr.scene.ngeom+=1
 return hr.render()
try:
 for f in range(total):
  i=min(round(f/fps*50),count-1);si=min(round((i+1)*2),len(a)-1);t=(i+1)*.02
  image=Image.new('RGB',size,'#101923');draw=ImageDraw.Draw(image)
  draw.text((15,10),f'Virtual Legs +10 cm | same recording / TWIST2 +500 | t={t:.2f}s',font=font20,fill='white')
  draw.text((15,38),'Left: original input    Right: support-level +10 cm, two-link IK, smooth swing gating',font=font16,fill='#c6d2dc')
  for col in range(2):
   x=col*540;label='ORIGINAL' if col==0 else '+10 cm + IK';color='#65b9f4' if col==0 else '#ffb15d'
   draw.text((x+15,65),label,font=font20,fill=color)
   image.paste(Image.fromarray(skeleton((raw if col==0 else patched)[si],raw[si])),(x,91))
   draw=ImageDraw.Draw(image);phase=int(a[si,2]);foot='none' if phase<0 else ['left','right'][phase]
   lift=a[si,3:5].sum()*100 if col else 0
   draw.text((x+15,371),f'1  Meta 14-point input | estimated swing: {foot} | added {lift:.1f} cm',font=font14,fill=color)
   q=sources[col]['qpos'][si];d.qpos[:7]=q[:7];d.qpos[qa]=q[7:];mujoco.mj_forward(m,d)
   base=sources[0]['qpos'][si];cam.lookat[:]=[base[0],base[1],.65];renderer.update_scene(d,camera=cam);image.paste(Image.fromarray(renderer.render()),(x,402))
   draw=ImageDraw.Draw(image);draw.text((x+15,682),'2  GMR retargeting (kinematic reference)',font=font16,fill=color)
   state=rolls[col]['state'][i];d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];mujoco.mj_forward(m,d);renderer.update_scene(d,camera=cam);image.paste(Image.fromarray(renderer.render()),(x,711))
   draw=ImageDraw.Draw(image);r=rows[col][i];draw.text((x+15,993),f'3  Physics | XY {r["root_xy_m"]*100:.0f} cm | slip {r["contact_slip_m_s"]*100:.1f} cm/s',font=font16,fill=color)
   if r['upward_contact_events']:draw.text((x+15,1017),'Contact candidate — not a confirmed trip',font=font14,fill='#ff7780')
  if f in [80,150,300,500,int(np.argmax(a[:,3:5].sum(axis=1))*.01*fps)]:image.save(out/f'preview-{f:04d}.png')
  proc.stdin.write(image.tobytes())
  if f%100==0:print(f,total,flush=True)
 proc.stdin.close();assert proc.wait()==0
finally:renderer.close();hr.close()

from pathlib import Path
import mujoco,numpy as np
from PIL import Image,ImageDraw,ImageFont
import subprocess,sys,json
O=Path(__file__).resolve().parent
S=O.parent/'rubik-fix-20261015'
m=mujoco.MjModel.from_xml_path(str(S/'assets-forward/scene-rubik.xml'))
m.vis.global_.offwidth=1920;m.vis.global_.offheight=1080
m.vis.headlight.active=1;m.vis.headlight.ambient[:]=.32;m.vis.headlight.diffuse[:]=.65
m.vis.quality.shadowsize=4096
m.stat.center[:]=[.3,0,.8];m.stat.extent=2.5
m.light_dir[0]=[.15,.2,-1]
m.vis.global_.fovy=38
D=mujoco.MjData(m)
D.qpos[:]=np.load(S/'rubik-spread-turn-hold-0.npz')['qpos'][0]
# Deliberately staged stationary pose for a title reveal, not a task rollout.
for side,sgn in [('left',1),('right',-1)]:
 for name,v in [('shoulder_pitch',-.8),('shoulder_roll',sgn*.2),('shoulder_yaw',0),('elbow',.65),('wrist_roll',0),('wrist_pitch',0),('wrist_yaw',0)]:
  try:D.qpos[m.joint(f'{side}_{name}_joint').qposadr[0]]=v
  except KeyError:pass
# Use the assembled cube's default pose, translated onto the tabletop.
D.qpos[50:]=m.qpos0[50:]
mujoco.mj_forward(m,D)
print('table',D.geom_xpos[m.geom('tabletop').id], 'cube',D.xpos[m.body('rubik_core').id],flush=True)
# Remove task target overlays and small pedestal; cube rests directly on table.
for i in range(m.ngeom):
 name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or ''
 if name in ('rubik_start_zone','rubik_stand'):m.geom_rgba[i,3]=0
cube_center=D.xpos[m.body('rubik_core').id].copy()
table_z=D.geom_xpos[m.geom('tabletop').id,2]+.025
shift=np.array([0,0,table_z+.051-cube_center[2]])
for j in range(m.njnt):
 if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE and m.jnt_qposadr[j]>=50:
  adr=m.jnt_qposadr[j];D.qpos[adr:adr+3]+=shift
mujoco.mj_forward(m,D)
print('shift',shift,'block0',D.geom_xpos[m.geom('rubik_block_0').id], 'block7',D.geom_xpos[m.geom('rubik_block_7').id],flush=True)
opt=mujoco.MjvOption();opt.geomgroup[3]=0
W,H=1280,720
r=mujoco.Renderer(m,H,W);cam=mujoco.MjvCamera();cam.azimuth=180;cam.distance=1.62

def smooth(x):
 x=np.clip(x,0,1);return x*x*(3-2*x)
def frame(t):
 u=smooth(t/6.5)
 cam.lookat[:]=[.12+.10*u,0,.18+.82*u]
 cam.elevation=-2-10*u
 r.update_scene(D,camera=cam,scene_option=opt)
 return Image.fromarray(r.render())
if '--preview' in sys.argv:
 for t in [0,3.25,6.5]:frame(t).save(O/f'preview-{t}.png')
 r.close();sys.exit()
FPS=30;duration=14.5
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',66)
small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',34)
p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(O/'rubik-coming-soon.mp4')],stdin=subprocess.PIPE)
held=frame(6.5)
for n in range(round(duration*FPS)):
 t=n/FPS
 im=frame(t) if t<6.5 else held.copy()
 fade=.75*smooth((t-8.3)/1.1)
 if fade:im=Image.blend(im,Image.new('RGB',(W,H)),float(fade))
 overlay=Image.new('RGBA',(W,H),(0,0,0,0))
 draw=ImageDraw.Draw(overlay)
 a=smooth((t-9.3)/.7)
 if a:draw.text((W/2,330),'coming soon',anchor='mm',font=font,fill=(245,245,245,round(255*a)))
 b=smooth((t-11.3)/.7)
 if b:draw.text((W/2,410),'(maybe)',anchor='mm',font=small,fill=(205,205,205,round(255*b)))
 im=Image.alpha_composite(im.convert('RGBA'),overlay).convert('RGB')
 p.stdin.write(np.asarray(im).tobytes())
 if n in [0,195,285,375]:im.save(O/f'frame-{n}.png')
p.stdin.close();assert p.wait()==0;r.close()
(O/'timing.json').write_text(json.dumps({'camera_rise':[0,6.5],'hold':[6.5,8.3],'dim_75_percent':[8.3,9.4],'coming_soon':[9.3,10],'maybe':[11.3,12],'end':duration,'note':'Staged offline MuJoCo render; static robot and cube, animated camera. Not a recorded task execution.'},indent=2))
print(O/'rubik-coming-soon.mp4')

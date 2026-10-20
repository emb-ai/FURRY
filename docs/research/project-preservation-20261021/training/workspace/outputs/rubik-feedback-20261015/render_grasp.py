from pathlib import Path
import numpy as np,mujoco
from PIL import Image,ImageDraw
p=Path(__file__).resolve().parent
z=dict(np.load(p/'coupled-series.npz'));m=mujoco.MjModel.from_xml_path(str(p.parent/'rubik-sequences-20261015/project/android/assets/scene-rubik.xml'));d=mujoco.MjData(m);r=mujoco.Renderer(m,500,600);o=mujoco.MjvOption();o.geomgroup[3]=0
im=Image.new('RGB',(1800,1030),'white');draw=ImageDraw.Draw(im)
for j,t in enumerate([40.5,120.3,131.9]):
 d.qpos[:]=z['qpos'][round(t*100)-1];mujoco.mj_forward(m,d)
 for i,az in enumerate([140,20]):
  c=mujoco.MjvCamera();c.lookat[:]=d.xpos[m.body('rubik_cubie_0').id];c.distance=.43;c.elevation=-25;c.azimuth=az;r.update_scene(d,camera=c,scene_option=o);im.paste(Image.fromarray(r.render()),(j*600,i*515));draw.text((j*600+12,i*515+5),f'{t:.2f}s',fill='black')
im.save(p/'grasp-comparison.png');r.close()

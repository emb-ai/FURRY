from pathlib import Path
import numpy as np,mujoco
from PIL import Image
p=Path(__file__).resolve().parent;z=np.load(p/'relative.npz');m=mujoco.MjModel.from_xml_path(str(p.parent/'rubik-sequences-20261015/project/android/assets/scene-rubik.xml'));d=mujoco.MjData(m);r=mujoco.Renderer(m,480,640);im=Image.new('RGB',(1920,960));opt=mujoco.MjvOption();opt.geomgroup[3]=0
for row,k in enumerate([1099,1289]):
 d.qpos[:]=z['qpos'][k];mujoco.mj_forward(m,d)
 for col,az in enumerate([90,180,270]):
  c=mujoco.MjvCamera();c.lookat[:]=[.36,0,1.];c.distance=.42;c.elevation=-20;c.azimuth=az;r.update_scene(d,camera=c,scene_option=opt);im.paste(Image.fromarray(r.render()),(col*640,row*480))
im.save(p/'grasp-views.png');r.close()

from pathlib import Path
import json,numpy as np,mujoco as mj
from PIL import Image,ImageDraw,ImageFont
O=Path(__file__).resolve().parents[1];D=Path(__file__).resolve().parent;cases=json.loads((O/'fresh-start/scenes.json').read_text())['cases'];ep=json.loads((O/'quality-audit.json').read_text())['accepted'][0];m=mj.MjModel.from_binary_path(str(O/'recordings'/ep/'learning-model.mjb'));d=mj.MjData(m);r=mj.Renderer(m,height=360,width=640);cam=m.camera('ego').id;m.light_castshadow[:]=0
im=Image.new('RGB',(960,700),'#101720');dr=ImageDraw.Draw(im);ft=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',19)
for col,i in enumerate([0,2,5]):
 c=cases[i];z=np.load(O/'fresh-start/finetuned'/f"act-{c['seed']}.npz")
 for row,t in enumerate([0,5,90]):
  j=min(len(z['qpos'])-1,max(0,np.searchsorted(z['trace'][:,0],t)-1));d.qpos[:]=c['qpos'] if t==0 else z['qpos'][j];d.mocap_pos.flat[:]=z['mocap_pos'];d.mocap_quat.flat[:]=z['mocap_quat'];mj.mj_forward(m,d);yaw,pitch=[0,0] if t==0 else z['actions'][j,37:39];cy,sy=np.cos(yaw),np.sin(yaw);cp,sp=np.cos(pitch),np.sin(pitch);R=d.cam_xmat[cam].reshape(3,3)@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])@np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]])
  r.update_scene(d,camera=cam)
  for cc in r.scene.camera:cc.pos[:]=d.cam_xpos[cam];cc.forward[:]=-R[:,2];cc.up[:]=R[:,1]
  r.scene.flags[mj.mjtRndFlag.mjRND_SHADOW]=0;image=Image.fromarray(r.render()).resize((224,224),Image.Resampling.BILINEAR);x=col*320;y=row*230;im.paste(image,(x+90,y));dr.text((x+3,y+5),f'#{i+1}\nt={min(t,z["trace"][-1,0]):.1f}c',font=ft,fill='white')
r.close();im.save(D/'ego-views.png')

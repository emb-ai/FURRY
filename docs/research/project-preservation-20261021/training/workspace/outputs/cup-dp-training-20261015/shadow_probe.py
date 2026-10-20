from pathlib import Path
import sys
from PIL import Image, ImageDraw
O=Path(__file__).resolve().parent
sys.path.insert(0,str(O.parent/'rubik-table-rest-20261015/project'))
from g1_sim.visual_dataset import inspect_episode,EpisodeRenderer
import mujoco
E=inspect_episode(O.parent/'task-demo-audit-20261015/recordings/episode-1792017537461911')
c=Image.new('RGB',(1280,770));d=ImageDraw.Draw(c)
for k,(name,extent,size,shadow) in enumerate([('Original',None,2048,True),('Shadow off',None,2048,False),('Focused shadow',2.5,4096,True),('High resolution',None,8192,True)]):
 r=EpisodeRenderer(E)
 r.renderer.close()
 if extent: r.model.vis.map.shadowclip=extent/r.model.stat.extent
 r.model.vis.quality.shadowsize=size
 if not shadow:r.model.light_castshadow[:]=0
 r.renderer=mujoco.Renderer(r.model,height=360,width=640)
 r.render(E['indices'][len(E['indices'])//2]);im=Image.fromarray(r.renderer.render());im.save(O/(name.replace(' ','-')+'.jpg'))
 x,y=k%2*640,k//2*385;c.paste(im,(x,y+25));d.text((x+10,y+5),name,fill='white');r.close()
c.save(O/'shadow-comparison.jpg')

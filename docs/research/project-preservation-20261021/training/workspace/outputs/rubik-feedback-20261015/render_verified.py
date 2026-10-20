from pathlib import Path
import numpy as np,mujoco,subprocess,shutil,os
from PIL import Image,ImageDraw,ImageFont
p=Path(__file__).resolve().parent;name=os.environ.get('RENDER_RUN','planned-five');start=float(os.environ.get('RENDER_START','117.01'));end=float(os.environ.get('RENDER_END','131.9'));label=os.environ.get('RENDER_LABEL','Пятый ход и возврат хвата');z=dict(np.load(p/(name+'.npz')));m=mujoco.MjModel.from_xml_path(str(p.parent/'rubik-sequences-20261015/project/android/assets/scene-rubik.xml'));d=mujoco.MjData(m);r=mujoco.Renderer(m,480,640);opt=mujoco.MjvOption();opt.geomgroup[3]=0;m.vis.headlight.ambient[:]=.25;m.vis.headlight.diffuse[:]=.5
cams=[]
for az in [140,20]:
 c=mujoco.MjvCamera();c.lookat[:]=[.38,.025,1.015];c.distance=.5;c.elevation=-25;c.azimuth=az;cams.append(c)
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',23);small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18);path=p/(name+'-verified.mp4');fps=25
proc=subprocess.Popen([shutil.which('ffmpeg'),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x640','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
for i in range(round((end-start)*fps)+50):
 t=min(end,start+i/fps);k=min(len(z['qpos'])-1,round(t*100)-1);d.qpos[:]=z['qpos'][k];mujoco.mj_forward(m,d);im=Image.new('RGB',(1280,640),'#14232b');draw=ImageDraw.Draw(im)
 for j,c in enumerate(cams):r.update_scene(d,camera=c,scene_option=opt);im.paste(Image.fromarray(r.render()),(640*j,45))
 draw.text((18,10),f'{label}   |   t = {t:.2f} с   |   скорость ×1',font=font,fill='white')
 draw.text((18,540),'Только кисти + обычный Grip. Кубик остаётся в руках, без опоры на стол.',font=font,fill='white')
 draw.text((18,577),'Пятый возврат: кубик 0,65 мм / 2,07° от эталона; звенья рук RMS 2,84 / 2,88 мм.',font=small,fill='#d9edee')
 draw.text((18,608),'Офлайн HOMIE Standing; исходная физика. Пять ходов одной грани; длинная серия ещё не принята.',font=small,fill='#c2d3df')
 proc.stdin.write(np.asarray(im).tobytes())
 if i==round((end-start)*fps):im.save(p/(name+'-verified-preview.png'))
proc.stdin.close();assert proc.wait()==0;r.close();subprocess.run([shutil.which('ffmpeg'),'-v','error','-i',str(path),'-f','null','-'],check=True);print(path)

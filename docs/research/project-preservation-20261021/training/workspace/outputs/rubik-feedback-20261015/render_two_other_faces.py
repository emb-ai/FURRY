from pathlib import Path
import numpy as np,mujoco,subprocess,shutil,os
from PIL import Image,ImageDraw,ImageFont
p=Path(__file__).resolve().parent;name=os.environ.get('RENDER_RUN','planned-five');start=float(os.environ.get('RENDER_START','117.01'));end=float(os.environ.get('RENDER_END','131.9'));label=os.environ.get('RENDER_LABEL','Пятый ход и возврат хвата');z=dict(np.load(p/(name+'.npz')));m=mujoco.MjModel.from_xml_path(str(p.parent/'rubik-sequences-20261015/project/android/assets/scene-rubik.xml'));d=mujoco.MjData(m);r=mujoco.Renderer(m,480,640);opt=mujoco.MjvOption();opt.geomgroup[3]=0;m.vis.headlight.ambient[:]=.25;m.vis.headlight.diffuse[:]=.5
cams=[]
for az in [140,20]:
 c=mujoco.MjvCamera();c.lookat[:]=[.38,.025,1.015];c.distance=.5;c.elevation=-25;c.azimuth=az;cams.append(c)
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',23);small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18);path=p/(name+'-verified.mp4');fps=25
proc=subprocess.Popen([shutil.which('ffmpeg'),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x640','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
timeline=[]
for a,b,speed in [(180.98,197.29,1),(197.29,203.29,1),(203.29,250.09,3),(250.09,254.75,1),(254.75,278.31,3),(278.31,283.67,1)]:
 timeline.extend((min(b,a+i*speed/fps),speed) for i in range(int(np.ceil((b-a)*fps/speed))))
timeline.extend([(283.67,1)]*50)
for i,(t,speed) in enumerate(timeline):
 k=min(len(z['qpos'])-1,round(t*100)-1);d.qpos[:]=z['qpos'][k];mujoco.mj_forward(m,d);im=Image.new('RGB',(1280,640),'#14232b');draw=ImageDraw.Draw(im)
 for j,c in enumerate(cams):r.update_scene(d,camera=c,scene_option=opt);im.paste(Image.fromarray(r.render()),(640*j,45))
 stageLabel=('1. Ход прежней грани и возврат' if t<=197.29 else '2. Поворот всего кубика' if t<=203.29 else '3. Перехват и коррекция хвата' if t<=250.09 else '4. Ход ДРУГОЙ грани' if t<=254.75 else '5. Возврат и коррекция хвата' if t<=278.31 else '6. Следующий ход новой грани');draw.text((18,10),f'{stageLabel}   |   t = {t:.2f} с   |   скорость ×{speed}',font=font,fill='white')
 draw.text((18,540),'Только кисти + обычный Grip. Кубик остаётся в руках, без опоры на стол.',font=font,fill='white')
 draw.text((18,577),'Прежняя грань: элементы 0, 4, 6, 7. После перехвата: элементы 1, 5, 6, 7.',font=small,fill='#d9edee')
 draw.text((18,608),'Офлайн HOMIE Standing. Продолжение после смены грани; точность повторяемого хвата ещё дорабатывается.',font=small,fill='#c2d3df')
 proc.stdin.write(np.asarray(im).tobytes())
 if i==len(timeline)-1:im.save(p/(name+'-verified-preview.png'))
proc.stdin.close();assert proc.wait()==0;r.close();subprocess.run([shutil.which('ffmpeg'),'-v','error','-i',str(path),'-f','null','-'],check=True);print(path)

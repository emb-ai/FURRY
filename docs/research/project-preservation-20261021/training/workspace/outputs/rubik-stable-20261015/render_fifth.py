from pathlib import Path
import numpy as np,mujoco,subprocess,shutil,os,json
from PIL import Image,ImageDraw,ImageFont
p=Path(__file__).resolve().parent;name=os.environ.get('RENDER_RUN','five-cycles');z=dict(np.load(p/(name+'.npz')));bounds=json.loads(os.environ.get('RENDER_BOUNDS','[117.01,139.59]'));count=len(bounds)-1;m=mujoco.MjModel.from_xml_path(str(p.parent/'rubik-sequences-20261015/project/android/assets/scene-rubik.xml'));d=mujoco.MjData(m);r=mujoco.Renderer(m,480,640);m.vis.headlight.ambient[:]=.25;m.vis.headlight.diffuse[:]=.5;opt=mujoco.MjvOption();opt.geomgroup[3]=0
cams=[]
for dist,el,az,at in [(.57,-25,140,[.38,.02,1.03]),(1.1,-18,160,[.23,0,1.02])]:
 c=mujoco.MjvCamera();c.lookat[:]=at;c.distance=dist;c.elevation=el;c.azimuth=az;cams.append(c)
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',23);small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18);path=p/(name+'-and-regrasp.mp4');fps=20;start=bounds[0];end=bounds[-1];speed=2
proc=subprocess.Popen([shutil.which('ffmpeg'),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x650','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
for frame in range(round((end-start)/speed*fps)+40):
 ts=min(end,start+frame/fps*speed);k=min(len(z['qpos'])-1,round(ts*100)-1);d.qpos[:]=z['qpos'][k];mujoco.mj_forward(m,d);im=Image.new('RGB',(1280,650),'#14232b');draw=ImageDraw.Draw(im)
 for h,c in enumerate(cams):r.update_scene(d,camera=c,scene_option=opt);im.paste(Image.fromarray(r.render()),(640*h,50))
 cycle=min(count,int(np.searchsorted(bounds,ts,side='right')))
 turning=ts<120.30
 free=z['commands'][k,-1]<.1 and z['trace'][k,13]==0
 phase='Поворот слоя' if turning else ('Рука открыта и отведена' if free else 'Перехват и восстановление положения пальцев')
 if ts>=end:phase='Пятый ход завершён; хват восстановлен'
 draw.text((18,12),f'{phase}  •  пятый ход  •  скорость ×2',font=font,fill='white')
 draw.text((18,540),'Только движения кистей и штатный Grip. Независимых команд пальцам нет.',font=font,fill='white')
 draw.text((18,577),'После пятого: RMS звеньев Л/П 6,4 / 3,9 мм. Кубик смещён на 43 мм и 20°.',font=small,fill='#d9edee')
 draw.text((18,612),'Офлайн MuJoCo + HOMIE Standing. Разные грани и длинная серия пока не приняты.',font=small,fill='#c2d3df')
 proc.stdin.write(np.asarray(im).tobytes())
 if abs(ts-139.4)<.06:im.save(p/(name+'-preview.png'))
proc.stdin.close();assert proc.wait()==0;r.close();subprocess.run([shutil.which('ffmpeg'),'-v','error','-i',str(path),'-f','null','-'],check=True);print(path)

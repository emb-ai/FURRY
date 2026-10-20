from pathlib import Path
import numpy as np,mujoco,subprocess,shutil,os,json
from PIL import Image,ImageDraw,ImageFont
p=Path(__file__).resolve().parent;name=os.environ.get('RENDER_RUN','three-cycles');z=dict(np.load(p/(name+'.npz')));bounds=json.loads(os.environ.get('RENDER_BOUNDS','[40.5,59.32,78.17,93.02]'));count=len(bounds)-1;m=mujoco.MjModel.from_xml_path(str(p.parent/'rubik-sequences-20261015/project/android/assets/scene-rubik.xml'));d=mujoco.MjData(m);r=mujoco.Renderer(m,480,640);m.vis.headlight.ambient[:]=.25;m.vis.headlight.diffuse[:]=.5;opt=mujoco.MjvOption();opt.geomgroup[3]=0
cams=[]
for dist,el,az,at in [(.57,-25,140,[.38,.02,1.03]),(1.1,-18,160,[.23,0,1.02])]:
 c=mujoco.MjvCamera();c.lookat[:]=at;c.distance=dist;c.elevation=el;c.azimuth=az;cams.append(c)
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',23);small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18);path=p/(name+'-and-regrasp.mp4');fps=20;start=bounds[0];end=bounds[-1];speed=2
proc=subprocess.Popen([shutil.which('ffmpeg'),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x650','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
for frame in range(round((end-start)/speed*fps)+40):
 ts=min(end,start+frame/fps*speed);k=min(len(z['qpos'])-1,round(ts*100)-1);d.qpos[:]=z['qpos'][k];mujoco.mj_forward(m,d);im=Image.new('RGB',(1280,650),'#14232b');draw=ImageDraw.Draw(im)
 for h,c in enumerate(cams):r.update_scene(d,camera=c,scene_option=opt);im.paste(Image.fromarray(r.render()),(640*h,50))
 cycle=min(count,int(np.searchsorted(bounds,ts,side='right')))
 turning=ts<43.92 or 59.32<=ts<62.77 or 78.17<=ts<82.52 or 93.02<=ts<96.31
 free=z['commands'][k,-1]<.1 and z['trace'][k,13]==0
 phase='Поворот слоя' if turning else ('Рука открыта и отведена' if free else 'Перехват и восстановление положения пальцев')
 if ts>=end:phase=f'{count} цикла завершены; хват восстановлен'
 draw.text((18,12),f'{phase}  •  цикл {cycle}/{count}  •  скорость ×2',font=font,fill='white')
 draw.text((18,540),f'{count} × 90° одной грани без сброса. После каждого — отпускание, отход и возврат.',font=font,fill='white')
 draw.text((18,577),'На границах циклов: смещение кубика 1–4 мм; RMS пальцев от эталона менее 8 мм.',font=small,fill='#d9edee')
 draw.text((18,612),'Офлайн MuJoCo + HOMIE Standing. Разные грани и устойчивое продолжение пока не подтверждены.',font=small,fill='#c2d3df')
 proc.stdin.write(np.asarray(im).tobytes())
 if abs(ts-72.1)<.06:im.save(p/(name+'-preview.png'))
proc.stdin.close();assert proc.wait()==0;r.close();subprocess.run([shutil.which('ffmpeg'),'-v','error','-i',str(path),'-f','null','-'],check=True);print(path)

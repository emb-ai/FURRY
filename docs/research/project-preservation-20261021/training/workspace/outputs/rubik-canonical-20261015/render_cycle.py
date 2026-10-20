from pathlib import Path
import numpy as np,mujoco,subprocess,shutil,os
from PIL import Image,ImageDraw,ImageFont
p=Path(__file__).resolve().parent;name=os.environ.get('RUN','macro-follow07');z=np.load(p/(name+'.npz'));m=mujoco.MjModel.from_xml_path(str(p.parent/'rubik-sequences-20261015/project/android/assets/scene-rubik.xml'));d=mujoco.MjData(m);r=mujoco.Renderer(m,480,640);m.vis.headlight.ambient[:]=.25;m.vis.headlight.diffuse[:]=.5;m.vis.headlight.specular[:]=.1;opt=mujoco.MjvOption();opt.geomgroup[3]=0
cams=[]
for distance,elevation,azimuth,lookat in [(.57,-22,140,[.36,0,1.04]),(1.1,-18,160,[.23,0,1.02])]:
 c=mujoco.MjvCamera();c.lookat[:]=lookat;c.distance=distance;c.elevation=elevation;c.azimuth=azimuth;cams.append(c)
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',23);small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18);path=p/(name+'-cycle.mp4');fps=20;end=float(os.environ.get('END','27.97'));start=10.8
proc=subprocess.Popen([shutil.which('ffmpeg'),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x600','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
for frame in range(round((end-start)*fps)):
 ts=start+frame/fps;k=min(len(z['qpos'])-1,round(ts*100)-1);d.qpos[:]=z['qpos'][k];mujoco.mj_forward(m,d);im=Image.new('RGB',(1280,600),'#14232b');draw=ImageDraw.Draw(im)
 for side,cam in enumerate(cams):r.update_scene(d,camera=cam,scene_option=opt);im.paste(Image.fromarray(r.render()),(640*side,45))
 free=z['commands'][k,-1]<.1 and z['trace'][k,13]==0;phase='Поворот слоя' if ts<14.97 else ('Правая рука отпущена и отведена' if free else 'Выравнивание / перехват')
 draw.text((18,10),f'{phase}   •   {ts-start:.1f} с',font=font,fill='white');draw.text((18,535),'Поворот → отпускание → отход → повторный захват. Один цикл; устойчивой серии пока нет.',font=font,fill='white');draw.text((18,570),'MuJoCo + HOMIE Standing, исходная физика. Офлайн воспроизведение; не запись экрана Quest.',font=small,fill='#c2d3df');proc.stdin.write(np.asarray(im).tobytes())
 if abs(ts-21.8)<.02:im.save(p/(name+'-release.png'))
proc.stdin.close();assert proc.wait()==0;r.close();subprocess.run([shutil.which('ffmpeg'),'-v','error','-i',str(path),'-f','null','-'],check=True);print(path)

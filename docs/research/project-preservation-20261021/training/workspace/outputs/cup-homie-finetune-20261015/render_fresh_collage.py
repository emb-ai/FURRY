from pathlib import Path
import json,subprocess,sys,hashlib
import numpy as np
import mujoco as mj
from PIL import Image,ImageDraw,ImageFont
O=Path(__file__).resolve().parent;F=O/'fresh-start';V=F/'video';V.mkdir(exist_ok=True)
summary=json.loads((F/'finetuned/act-summary.json').read_text());cases=json.loads((F/'scenes.json').read_text())['cases'];assert summary['n']==10 and not any(r['error'] for r in summary['results'])
episode=json.loads((O/'quality-audit.json').read_text())['accepted'][0]
m=mj.MjModel.from_binary_path(str(O/'recordings'/episode/'learning-model.mjb'));d=mj.MjData(m);m.light_castshadow[:]=0
r=mj.Renderer(m,height=340,width=480)
cam=mj.MjvCamera();cam.lookat[:]=[.25,0,.75];cam.distance=1.9;cam.azimuth=135;cam.elevation=-25
font='/System/Library/Fonts/Supplemental/Arial.ttf';ft=ImageFont.truetype(font,20);small=ImageFont.truetype(font,17);big=ImageFont.truetype(font,30)
clips=[]
for case,row in zip(cases,summary['results']):
 assert case['seed']==row['seed'];p=F/'finetuned'/f"act-{case['seed']}.npz";a=np.load(p);clips.append((case,row,a))
W,H=2400,930;fps=15;speed=2;duration=90

def frame(t):
 im=Image.new('RGB',(W,H),(16,23,32));draw=ImageDraw.Draw(im)
 draw.text((20,12),'ACT · 10 случайных сред · полный запуск с нуля · успех 1/10',font=big,fill='white')
 draw.text((20,52),f'Время симуляции: {t:05.1f} / 90 с  |  Скорость видео: ×2  |  Восстановление сохранённых состояний теста',font=ft,fill='#b7c5d4')
 for i,(c,row,a) in enumerate(clips):
  x=(i%5)*480;y=90+(i//5)*410;times=a['trace'][:,0];j=max(0,min(int(np.searchsorted(times,t,side='right')-1),len(times)-1))
  d.qpos[:]=c['qpos'] if t==0 else a['qpos'][j];d.qvel[:]=0;d.mocap_pos.flat[:]=a['mocap_pos'];d.mocap_quat.flat[:]=a['mocap_quat'];mj.mj_forward(m,d)
  r.update_scene(d,camera=cam);r.scene.flags[mj.mjtRndFlag.mjRND_SHADOW]=0
  im.paste(Image.fromarray(r.render()),(x,y+38));draw=ImageDraw.Draw(im)
  color='#69e6a6' if row['success'] else '#f6b86e'
  draw.text((x+12,y+8),f"{i+1:02d} · seed {c['seed']}",font=ft,fill='white')
  done=t+1e-6>=row['sim_s'];label=('УСПЕХ' if row['success'] else 'ТАЙМАУТ') if done else 'выполняется'
  if done and row['final_cup_tilt_deg']>70:label+=' · чашка на боку'
  draw.text((x+12,y+380),label,font=small,fill=color if done else '#b7c5d4')
  draw.rectangle((x,y,x+479,y+409),outline='#344355',width=2)
 draw.text((20,918-10),'Одинаковая камера. После завершения сцена замирает. Это replay результатов симуляции, не запись из шлема.',font=small,fill='#b7c5d4')
 return im
if '--final' in sys.argv:
 frame(90).save(V/'final-results.png');r.close();sys.exit()
frame(0).save(V/'act-collage-preview.png')
if '--preview' in sys.argv:r.close();sys.exit()
video=V/'act-full-reset-10-collage.mp4';proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
N=int(duration/speed*fps)+1
for k in range(N):
 im=frame(min(duration,k/fps*speed));proc.stdin.write(im.tobytes())
 if k in [90,300,675]:im.save(V/f'preview-{k}.png')
 if k%75==0:print(json.dumps({'frame':k,'total':N}),flush=True)
proc.stdin.close();assert proc.wait()==0;r.close()
(V/'provenance.json').write_text(json.dumps({'source':str(F/'finetuned/act-summary.json'),'protocol':'native full reset; no demo prefix','n':10,'successes':1,'sim_seconds':90,'speed':speed,'fps':fps,'frames':N,'resolution':[W,H],'method':'Render stored evaluated qpos with original model and per-scene goal; no new policy rollout. Finished trials freeze.','source_sha256':{str(F/'finetuned'/f"act-{c['seed']}.npz"):hashlib.sha256((F/'finetuned'/f"act-{c['seed']}.npz").read_bytes()).hexdigest() for c in cases}},indent=2))
print(str(video),flush=True)

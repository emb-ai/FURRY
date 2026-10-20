from pathlib import Path
import json,subprocess,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
O=Path(__file__).resolve().parent
W,H,FPS=3840,2160,25
names=['TWIST2 исходная','TWIST2 дообученная','TWIST2 подъём стоп','TWIST2 узкая стойка','HOMIE ходьба','DexWeave широкая стойка','DexWeave узкая стойка','SONIC ходьба']
colors=['#72d4e5','#ffb47a','#c9b0ff','#ff99af','#81dbc1','#b2dc82','#e1c078','#8bb2ff']
metrics=json.loads((O/'metrics.json').read_text())['results'];values=np.array([[r['mean_xy_cm'],r['mean_yaw_deg']] for r in metrics]);lo=values.min(0);hi=values.max(0)
font='/System/Library/Fonts/Supplemental/Arial'
F=lambda size,bold=False:ImageFont.truetype(font+(' Bold' if bold else '')+'.ttf',size)
title=F(64,True);label=F(44,True);desc=F(36);number=F(112,True)
# Explicit fit checks before exporting.
for s in names:assert title.getlength(s)<924,(s,title.getlength(s))
for row in values:
 for s in [f'{row[0]:.1f} см',f'{row[1]:.1f}°']:assert number.getlength(s)<430,(s,number.getlength(s))
def titles(im):
 im=im.convert('RGB');d=ImageDraw.Draw(im)
 for n,s in enumerate(names):
  x=(n%4)*960;y=(n//4)*1080
  d.rectangle((x+2,y+2,x+957,y+99),fill='#101a22')
  d.text((x+480,y+50),s,font=title,fill=colors[n],anchor='mm')
 return im
base=titles(Image.open(O/'last-frame.png')).convert('RGBA')
def overlay(alpha=1):
 layer=Image.new('RGBA',base.size);d=ImageDraw.Draw(layer)
 for n,r in enumerate(metrics):
  x=(n%4)*960;y=(n//4)*1080
  for j,(heading,detail,unit) in enumerate([('Ошибка позиции','MAE',' см'),('Ошибка поворота','MAE','°')]):
   px=x+24+j*464;py=y+504;w=448;h=275
   p=(values[n,j]-lo[j])/max(hi[j]-lo[j],1e-10);a=np.array([25,113,95]);b=np.array([163,54,46]);c=tuple(np.rint(a*(1-p)+b*p).astype(int))
   d.rounded_rectangle((px,py,px+w,py+h),radius=20,fill=(*c,222))
   d.text((px+w/2,py+22),heading,font=label,fill='white',anchor='mt')
   d.text((px+w/2,py+77),detail,font=desc,fill='#f0f5f7',anchor='mt')
   d.text((px+w/2,py+142),f'{values[n,j]:.1f}{unit}',font=number,fill='white',anchor='mt')
 if alpha<1:layer.putalpha(layer.getchannel('A').point(lambda x:int(x*alpha)))
 return Image.alpha_composite(base,layer).convert('RGB')
titles(Image.open(O/'preview.png')).save(O/'preview-v3.png');overlay().save(O/'heatmap-preview-v3.png')
(O/'display-labels-v3.json').write_text(json.dumps(dict(names=names,title_font=64,metric_label_font=44,metric_definition_font=36,metric_value_font=112,position_metric='Mean Euclidean XY distance: mean(sqrt(dx^2+dy^2)), not coordinate-wise MAE and not RMSE',rotation_metric='Yaw MAE: mean(abs(wrap(yaw_robot-yaw_reference)))'),ensure_ascii=False,indent=2))
if '--preview' in sys.argv:sys.exit()
video=O/'eight-policies-right-turn-v3.mp4'
dec=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-i',str(O/'eight-policies-right-turn.mp4'),'-t','20','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE)
enc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r','25','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
try:
 for k in range(500):
  b=dec.stdout.read(W*H*3);assert len(b)==W*H*3,(k,len(b))
  enc.stdin.write(titles(Image.frombytes('RGB',(W,H),b)).tobytes())
  if k%100==0:print('titles',k,'/500',flush=True)
 assert not dec.stdout.read(1);assert dec.wait()==0
 for k in range(150):enc.stdin.write(overlay(min(1,(k+1)/20)).tobytes())
 enc.stdin.close();assert enc.wait()==0
finally:
 for p in [dec,enc]:
  if p.poll() is None:p.kill();p.wait()
print(video,flush=True)

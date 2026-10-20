from pathlib import Path
import json,subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
O=Path(__file__).resolve().parent
base=Image.open(O/'last-frame.png').convert('RGBA');W,H=base.size
metrics=json.loads((O/'metrics.json').read_text())['results'];values=np.array([[r['mean_xy_cm'],r['mean_yaw_deg']] for r in metrics]);lo=values.min(0);hi=values.max(0)
font='/System/Library/Fonts/Supplemental/Arial'
label=ImageFont.truetype(font+'.ttf',30);number=ImageFont.truetype(font+' Bold.ttf',68)
def overlay(alpha=1):
 layer=Image.new('RGBA',base.size);d=ImageDraw.Draw(layer)
 for n,r in enumerate(metrics):
  x=(n%4)*960;y=(n//4)*1080
  for j,(title,unit) in enumerate([('Ошибка позиции','см'),('Ошибка поворота','°')]):
   px=x+30+j*460;py=y+560;w=440;h=180
   p=(values[n,j]-lo[j])/max(hi[j]-lo[j],1e-10)
   a=np.array([35,125,106]);b=np.array([176,65,54]);c=tuple(np.rint(a*(1-p)+b*p).astype(int))
   d.rounded_rectangle((px,py,px+w,py+h),radius=20,fill=(*c,205))
   d.text((px+w/2,py+21),title,font=label,fill='#f0f5f7',anchor='mt')
   d.text((px+w/2,py+87),f'{values[n,j]:.1f} {unit}',font=number,fill='white',anchor='mt')
 if alpha<1:layer.putalpha(layer.getchannel('A').point(lambda x:int(x*alpha)))
 return Image.alpha_composite(base,layer).convert('RGB')
overlay().save(O/'heatmap-preview.png')
end=O/'per-cell-metrics.mp4'
p=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r','25','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(end)],stdin=subprocess.PIPE)
for f in range(150):p.stdin.write(overlay(min(1,(f+1)/20)).tobytes())
p.stdin.close();assert p.wait()==0
# Preserve the already rendered 20-second motion, replacing only its ending.
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(O/'eight-policies-right-turn.mp4'),'-t','20','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p',str(O/'motion-only.mp4')],check=True)
(O/'concat.txt').write_text("file 'motion-only.mp4'\nfile 'per-cell-metrics.mp4'\n")
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(O/'concat.txt'),'-c','copy','-movflags','+faststart',str(O/'eight-policies-right-turn-v2.mp4')],check=True)
print(O/'eight-policies-right-turn-v2.mp4',flush=True)

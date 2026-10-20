from pathlib import Path
import json,subprocess,sys
import numpy as np
import mujoco as mj
from PIL import Image,ImageDraw,ImageFont
O=Path(__file__).resolve().parent;P=O.parents[1]/'outputs/training-capture-20261015/project';A=P/'android/assets'
protocol=json.loads((O/'protocol.json').read_text());vs=protocol['variants'];z=np.load(O/'references.npz');times=z['time'];ref=z['base'];raw=[np.loadtxt(O/(v['id']+'.csv'),delimiter=',') for v in vs]
assert all(v['exit_code']==0 for v in vs),[(v['id'],v['exit_code']) for v in vs]
assert all(len(a)==2001 and abs(a[-1,0]-20)<1e-6 and np.isfinite(a).all() for a in raw)
qs=[a[:,1:] for a in raw]
assert all(np.array_equal(qs[0][0],q[0]) for q in qs)
def yaw(q):return np.arctan2(2*(q[:,3]*q[:,6]+q[:,4]*q[:,5]),1-2*(q[:,5]**2+q[:,6]**2))
metrics=[]
for v,q in zip(vs,qs):
 xy=np.linalg.norm(q[1:,:2]-ref[1:,:2],axis=1);dy=yaw(q[1:])-yaw(ref[1:]);angle=np.abs(np.arctan2(np.sin(dy),np.cos(dy)))
 metrics.append(dict(id=v['id'],name=v['title'],mean_xy_cm=float(xy.mean()*100),mean_yaw_deg=float(np.rad2deg(angle).mean()),fall=False,duration_s=20))
(O/'metrics.json').write_text(json.dumps(dict(definition=protocol['metric'],results=metrics),indent=2))
W,H=3840,2160;CW,CH=960,1080;FPS=25;RH=720;RW=944
fontpath='/System/Library/Fonts/Supplemental/Arial'
F=lambda s,b=False:ImageFont.truetype(fontpath+(' Bold' if b else '')+'.ttf',s)
fonts={s:F(s) for s in [26,30,34,38,42,48]};bold={s:F(s,True) for s in [38,42,48,56]}
colors=['#72d4e5','#ffb47a','#c9b0ff','#ff99af','#81dbc1','#b2dc82','#e1c078','#8bb2ff'];bg='#101a22';muted='#bec7ce'
m=mj.MjModel.from_xml_path(str(A/'scene-stand.xml'));data=mj.MjData(m);renderer=mj.Renderer(m,height=RH,width=RW)
cam=mj.MjvCamera();cam.distance=2.55;cam.elevation=-14;cam.azimuth=135
xy=np.concatenate([ref[:,:2],*[q[:,:2] for q in qs]]);center=(xy.min(0)+xy.max(0))/2;span=xy.max(0)-xy.min(0);scale=min(825/(span[0]+.6),235/(span[1]+.6))

def frame(k):
 canvas=Image.new('RGB',(W,H),bg)
 for n,(v,q) in enumerate(zip(vs,qs)):
  x=(n%4)*CW;y=(n//4)*CH;color=colors[n];a=q[k];data.qpos[:]=a;mj.mj_forward(m,data);cam.lookat[:]=[a[0],a[1],.72]
  renderer.update_scene(data,camera=cam);canvas.paste(Image.fromarray(renderer.render()),(x+8,y+73))
  d=ImageDraw.Draw(canvas);d.text((x+CW/2,y+14),v['title'],font=bold[42],fill=color,anchor='mt')
  d.rounded_rectangle((x+16,y+806,x+944,y+1063),radius=10,fill='#1a2832')
  def pts(p):return [(float(x+CW/2+(v[0]-center[0])*scale),float(y+932-(v[1]-center[1])*scale)) for v in p]
  d.line(pts(ref[::4,:2]),fill='#9da8b2',width=3)
  if k>0:d.line(pts(q[:k+1:2,:2]),fill=color,width=5)
  for p,c in [(pts(ref[k:k+1,:2])[0],muted),(pts(a[None,:2])[0],color)]:d.ellipse((p[0]-7,p[1]-7,p[0]+7,p[1]+7),fill=c)
  # Unlabelled scale bar: presentation only contains the policy title.
  if n%4:d.line((x,y+8,x,y+CH-8),fill='#31414d',width=2)
  if n>=4:d.line((x+8,y,x+CW-8,y),fill='#31414d',width=2)
 return canvas

def heatmap(base,alpha=1):
 overlay=Image.new('RGBA',(W,H));d=ImageDraw.Draw(overlay)
 left,top,right,bottom=1010,300,2830,1830
 d.rounded_rectangle((left,top,right,bottom),radius=36,fill=(10,19,28,220))
 d.text((W/2,top+56),'Средняя ошибка · 20 с',font=bold[56],fill='#f2f6f8',anchor='mt')
 colx=[1900,2370];cw=390;rowtop=top+255;rowh=140
 d.text((left+70,top+175),'Политика',font=bold[38],fill='#e3ebf0')
 for x,label in zip(colx,['Позиция, см','Поворот, °']):d.text((x+cw/2,top+175),label,font=bold[38],fill='#e3ebf0',anchor='mt')
 vals=np.array([[r['mean_xy_cm'],r['mean_yaw_deg']] for r in metrics]);lo=vals.min(0);hi=vals.max(0)
 for n,r in enumerate(metrics):
  y=rowtop+n*rowh
  d.text((left+70,y+49),r['name'],font=bold[38],fill=colors[n])
  for j,x in enumerate(colx):
   p=(vals[n,j]-lo[j])/max(hi[j]-lo[j],1e-10);a=np.array([45,137,119]);b=np.array([186,85,69]);c=tuple(np.rint(a*(1-p)+b*p).astype(int))
   d.rounded_rectangle((x,y+9,x+cw,y+rowh-9),radius=14,fill=(*c,208))
   d.text((x+cw/2,y+rowh/2),f'{vals[n,j]:.1f}',font=bold[48],fill='white',anchor='mm')
 d.text((W/2,bottom-58),'Меньше — лучше',font=fonts[34],fill='#dce6ec',anchor='mm')
 if alpha<1:overlay.putalpha(overlay.getchannel('A').point(lambda x:int(x*alpha)))
 return Image.alpha_composite(base.convert('RGBA'),overlay).convert('RGB')

if '--preview' in sys.argv:
 frame(1500).save(O/'preview.png');heatmap(frame(2000)).save(O/'heatmap-preview.png');renderer.close();print(json.dumps(metrics,indent=2));sys.exit()
video=O/'eight-policies-right-turn.mp4'
proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
try:
 for f in range(500):
  im=frame(f*4)
  if f==375:im.save(O/'preview.png')
  proc.stdin.write(im.tobytes())
  if f%100==0:print('render',f,'/500',flush=True)
 last=frame(2000);last.save(O/'last-frame.png');heatmap(last).save(O/'heatmap-preview.png')
 for f in range(150):proc.stdin.write(heatmap(last,min(1,(f+1)/20)).tobytes())
 proc.stdin.close();assert proc.wait()==0
finally:
 renderer.close()
 if proc.poll() is None:proc.kill();proc.wait()
print(video,flush=True)

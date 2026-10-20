from pathlib import Path
import subprocess,json,concurrent.futures,shutil
from PIL import Image,ImageDraw,ImageFont
base=Path(__file__).resolve().parent;sources=json.loads((base/'sources.json').read_text());out=base/'continuous-returns';out.mkdir(exist_ok=True)
clips=[(9,7.7,12.4,'Со стороны · IMG_3699 · 00:07.7–00:12.4'),(8,173.1,175.3,'Со стороны · IMG_3698 · 02:53.1–02:55.3'),(3,36.8,45.8,'Из очков: соперник · 16:17:26 · 00:36.8–00:45.8'),(2,120.8,123.0,'Симуляция · возврат через сетку · 02:00.8–02:03.0'),(2,135.9,137.8,'Симуляция · отскок вверх · 02:15.9–02:17.8'),(2,150.9,153.1,'Симуляция · мяч к левому краю сетки · 02:30.9–02:33.1'),(1,47.6,55.0,'Симуляция · два возврата подряд · 00:47.6–00:55.0')]
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',25)
def run(item):
 i,(id,start,end,label)=item
 if i < 6:
  shutil.copy2(base/'three-hits'/f'cut-{i}.mp4',out/f'cut-{i}.mp4')
  return {'index':i,'source':sources[id]['path'],'in':start,'out':end,'label':label}
 banner=Image.new('RGB',(1280,64),(16,20,26));d=ImageDraw.Draw(banner);d.text((24,18),label,font=font,fill='white');banner.save(out/f'label-{i}.png')
 cmd=['ffmpeg','-y','-v','error','-ss',str(start),'-i',sources[id]['path'],'-loop','1','-i',str(out/f'label-{i}.png'),'-t',str(end-start),'-filter_complex','[0:v]fps=30,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:784:(ow-iw)/2:0:color=0x10141a,setsar=1[v];[v][1:v]overlay=0:720,format=yuv420p[out]','-map','[out]','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','19','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-movflags','+faststart',str(out/f'cut-{i}.mp4')]
 subprocess.run(cmd,check=True)
 return {'index':i,'source':sources[id]['path'],'in':start,'out':end,'label':label}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool: selected=list(pool.map(run,enumerate(clips)))
(out/'selection.json').write_text(json.dumps({'clips':selected,'note':'Editorial selection of separate takes, not synchronized views; Visual frame review: contact near 121.7s followed by ball crossing net; contact near 136.8s sends ball upward, landing not confirmed. Third contact near 151.8s sends ball toward left edge of net and off left side; legal landing not confirmed. Three contacts, not three confirmed legal returns. Replaced incorrect three-contact ending with one continuous excerpt 47.6–55.0s from operator-20261015-152710-107. Contacts around 48.4s and 53.4s, outbound flight and distant bounce visually reviewed at 15 fps. No sequence of three successful consecutive returns confirmed. Original order and speed preserved. Silent review draft; original speed.'},ensure_ascii=False,indent=2))
(out/'concat.txt').write_text(''.join(f"file 'cut-{i}.mp4'\n" for i in range(len(clips))))
subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(out/'concat.txt'),'-c','copy','-movflags','+faststart',str(out/'tennis-montage-v7.mp4')],check=True)
subprocess.run(['ffmpeg','-v','error','-i',str(out/'tennis-montage-v7.mp4'),'-f','null','-'],check=True)
print(out/'tennis-montage-v7.mp4')

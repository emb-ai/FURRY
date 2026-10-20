from pathlib import Path
import json,subprocess,concurrent.futures,shutil
from PIL import Image,ImageDraw,ImageFont
base=Path(__file__).resolve().parent;out=base/'combined-v2';out.mkdir(exist_ok=True)
ten=base.parent/'tennis-edit-20261016/continuous-returns'
clips=[
(base.parent/'cup-dp-training-20261015/cup-no-shadow.mp4',0,269/30,'Чашка: перенос по столу · replay записи оператора'),
(Path('/Users/vlad/Downloads/Telegram Desktop/org.g1lab.quest.smooth-20261015-103702-0.mp4'),101.0,119.0,'Kick-T: совмещение с целью · запись из очков · 01:41–01:59'),
(Path('/Users/vlad/Downloads/org.g1lab.quest-20261015-114917-0.mp4'),9.0,13.8,'Sixseven · оператор и робот · запись из очков')]

font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',25)
def run(item):
 i,(src,start,end,label)=item
 if i == 0:
  shutil.copy2(base/'combined/cut-0.mp4',out/'cut-0.mp4')
  return {'source':str(src),'in':start,'out':end,'label':label,'index':i}
 banner=Image.new('RGB',(1280,64),(16,20,26));ImageDraw.Draw(banner).text((24,18),label,font=font,fill='white');banner.save(out/f'label-{i}.png')
 filt='[0:v]fps=30,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:784:(ow-iw)/2:0:color=0x10141a,setsar=1[v];[v][1:v]overlay=0:720,format=yuv420p[out]'
 subprocess.run(['ffmpeg','-y','-v','error','-ss',str(start),'-i',str(src),'-loop','1','-i',str(out/f'label-{i}.png'),'-t',str(end-start),'-filter_complex',filt,'-map','[out]','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','19','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-movflags','+faststart',str(out/f'cut-{i}.mp4')],check=True)
 return {'source':str(src),'in':start,'out':end,'label':label,'index':i}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:selection=list(pool.map(run,enumerate(clips)))
old=json.loads((ten/'selection.json').read_text())
for i,c in enumerate(old['clips'],3):
 shutil.copy2(ten/f"cut-{c['index']}.mp4",out/f'cut-{i}.mp4');selection.append(dict(c,index=i))
(out/'selection.json').write_text(json.dumps({'clips':selection,'note':'Cup is offline render of recorded operator demonstration. T clips are headset screen recordings; T selection is the final trajectory from 103702, with the closest visually observed alignment; no numeric success metric inferred. Sixseven shows alternating operator and robot hands from 114917. Tennis selection is unchanged from v7. Silent review draft, original speed. Box placement not included: no verified completed video found.'},ensure_ascii=False,indent=2))
(out/'concat.txt').write_text(''.join(f"file 'cut-{i}.mp4'\n" for i in range(len(selection))))
video=out/'cup-kick-t-sixseven-tennis.mp4'
subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(out/'concat.txt'),'-c','copy','-movflags','+faststart',str(video)],check=True)
subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True)
print(video)

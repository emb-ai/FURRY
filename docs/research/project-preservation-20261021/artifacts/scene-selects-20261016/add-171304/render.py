from pathlib import Path
import subprocess,json,concurrent.futures
p=Path(__file__).resolve().parent
src=Path('/Users/vlad/Downloads/2026-10-08 17-13-04.mp4')
previous=p.parent/'combined-v2/cup-kick-t-sixseven-tennis-no-captions.mp4'
clips=[(16.2,24.8),(39.0,43.8)]
def render(item):
 i,(start,end)=item
 dst=p/f'new-{i}.mp4'
 subprocess.run(['ffmpeg','-y','-v','error','-ss',str(start),'-i',str(src),'-t',str(end-start),'-vf','crop=1920:1040:0:40,fps=30,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-video_track_timescale','15360','-movflags','+faststart',str(dst)],check=True)
 return dst
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:parts=list(ex.map(render,enumerate(clips)))
parts.append(previous)
(p/'concat.txt').write_text(''.join("file '"+str(f)+"'\n" for f in parts))
dst=p/'tracking-cup-kick-t-sixseven-tennis.mp4'
subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(p/'concat.txt'),'-c','copy','-movflags','+faststart',str(dst)],check=True)
subprocess.run(['ffmpeg','-v','error','-i',str(dst),'-f','null','-'],check=True)
(p/'selection.json').write_text(json.dumps({'new_clips':[{'source':str(src),'in':a,'out':b} for a,b in clips],'followed_by':str(previous),'note':'Added two real camera / 3D marker tracking excerpts before existing montage. Original speed. Removed application title bar only. No added captions; silent.'},ensure_ascii=False,indent=2))
print(dst)

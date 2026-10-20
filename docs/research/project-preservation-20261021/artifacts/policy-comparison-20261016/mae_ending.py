from pathlib import Path
O=Path(__file__).resolve().parent
source=(O/'readable_overlay.py').read_text().split("if '--preview' in sys.argv:")[0]
exec(compile(source,str(O/'readable_overlay.py'),'exec'))
end=O/'mae-ending.mp4'
p=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r','25','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',str(end)],stdin=subprocess.PIPE)
for k in range(150):p.stdin.write(overlay(min(1,(k+1)/20)).tobytes())
p.stdin.close();assert p.wait()==0
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(O/'eight-policies-right-turn-v3.mp4'),'-t','20','-an','-c','copy',str(O/'motion-v3.mp4')],check=True)
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=nb_frames:format=duration','-of','json',str(O/'motion-v3.mp4')]))
assert int(probe['streams'][0]['nb_frames'])==500 and abs(float(probe['format']['duration'])-20)<.001,probe
(O/'concat-v4.txt').write_text("file 'motion-v3.mp4'\nfile 'mae-ending.mp4'\n")
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(O/'concat-v4.txt'),'-c','copy','-movflags','+faststart',str(O/'eight-policies-right-turn-v4.mp4')],check=True)
print(O/'eight-policies-right-turn-v4.mp4',flush=True)

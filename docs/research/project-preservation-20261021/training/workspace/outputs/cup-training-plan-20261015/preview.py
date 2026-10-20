from pathlib import Path
import sys,json,numpy as np
from PIL import Image,ImageDraw,ImageFont
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O.parent/'rubik-table-rest-20261015/project'))
from g1_sim.visual_dataset import inspect_episode,EpisodeRenderer
R=O.parent/'task-demo-audit-20261015/recordings';a=json.loads((O/'audit.json').read_text());good=[r for r in a['episodes'] if r['accepted']];chosen=[good[0],good[13],*[r for r in good[14:] if r['layout']['distance_m']>=.32][:3],good[-1]]
canvas=Image.new('RGB',(896,250*len(chosen)),'#12202c');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',14)
for row,r in enumerate(chosen):
 e=inspect_episode(R/r['episode']);renderer=EpisodeRenderer(e)
 try:
  for col,fraction in enumerate([0,.33,.67,1]):
   idx=e['indices'][round((len(e['indices'])-1)*fraction)];canvas.paste(Image.fromarray(renderer.render(idx)),(224*col,250*row+26))
 finally:renderer.close()
 draw.text((8,250*row+5),r['episode']+' | transfer '+str(round(r['layout']['distance_m']*100))+' cm | 0 / 33 / 67 / 100%',font=font,fill='white')
canvas.save(O/'training-rgb-preview.jpg');(O/'preview-episodes.json').write_text(json.dumps(chosen,indent=2));print('Preview saved')

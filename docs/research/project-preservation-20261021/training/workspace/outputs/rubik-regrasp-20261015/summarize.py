from pathlib import Path
import json
p=Path(__file__).resolve().parent
for f in sorted(p.glob('*.json')):
 try:ev=json.loads(f.read_text());seq=[e for e in ev if e['kind']=='settled']
 except (TypeError,KeyError):continue
 if seq:print(f.stem,len(seq),'z',round(seq[-1]['cube'][2],3),'finger',[[round(v,1) for v in e['finger_rms_mm']] for e in seq], 'contacts',[e['contacts'] for e in seq])

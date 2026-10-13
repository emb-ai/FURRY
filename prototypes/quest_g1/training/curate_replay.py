"""Choose replay fixtures by baseline success, without consulting Quest validation."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);p.add_argument('baseline',type=Path);a=p.parse_args();r=json.loads((a.dataset/'manifest.json').read_text());b=json.loads(a.baseline.read_text());passed={t['clip'] for t in b['trials'] if t['stage']==0 and not t['fell'] and t['seed']==0}
if not passed:raise ValueError('No eligible baseline replay clips')
r['replay_eligibility']={'criterion':'full baseline survival on same MuJoCo embodiment, seed 0; no Quest validation used','kept':sorted(passed),'excluded':[c['id'] for c in r['clips'] if c['split']=='replay' and c['id'] not in passed]};r['clips']=[c for c in r['clips'] if c['split']!='replay' or c['id'] in passed];(a.dataset/'manifest.json').write_text(json.dumps(r,indent=2));print(r['replay_eligibility'])

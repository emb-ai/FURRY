import json
import os
from pathlib import Path

code=Path(__file__).resolve().parent
root=code.parent
report=[]
for row in json.loads((code/'submitted.json').read_text())['runs']:
    path=root/'runs'/row['run_id']
    try:os.kill(row['pid'],0);alive=True
    except ProcessLookupError:alive=False
    metrics=path/'training/metrics.jsonl'
    rows=[]
    if metrics.exists():
        for line in metrics.read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:pass
    verdicts=[]
    for p in sorted(path.glob('checkpoint_*.gait-verdict.json')):
        v=json.loads(p.read_text());verdicts.append({'checkpoint':p.stem,**v})
    report.append({'run_id':row['run_id'],'gpu':row['gpu'],'pipeline_alive':alive,'last_training_record':rows[-1] if rows else None,'checkpoints_evaluated':len(verdicts),'pipeline_complete':(path/'result.json').exists(),'last_gait_verdict':verdicts[-1] if verdicts else None,'log_tail':(code/(row['run_id']+'.log')).read_text(errors='replace').splitlines()[-5:] if not alive else []})
out={'runs':report}
(code/'status.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))

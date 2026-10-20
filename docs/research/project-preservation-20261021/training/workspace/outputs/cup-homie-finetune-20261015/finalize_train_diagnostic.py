from pathlib import Path
import json,time
p=Path(__file__).resolve().parent/'train-diagnostic'
for _ in range(180):
 d=json.loads((p/'status.json').read_text())
 if d['phase'] in ['completed','failed']:
  for r in d['results']:
   if r.get('mode')=='act':
    f=p/f"act-new-train-h8-{r['episode']}.json";f.write_text(json.dumps(r,indent=2))
  if d['phase']=='completed':
   report={'summary':d['summary'],'act_source_step':100000,'act_finetune_updates':20000,'protocol':'Eight fixed train episodes; initial physical snapshot restored; HOMIE weights loaded, controller history starts empty','causality':'Mismatch in recorded-command replay is confirmed; missing controller snapshot is not established as its cause','observation_action_audit':json.loads((p/'observation-action-audit.json').read_text())}
   (p/'report.json').write_text(json.dumps(report,indent=2))
  break
 time.sleep(10)

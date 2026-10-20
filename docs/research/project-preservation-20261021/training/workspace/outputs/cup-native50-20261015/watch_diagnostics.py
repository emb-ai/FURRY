from pathlib import Path
import os,sys,time,json,subprocess
O=Path(__file__).resolve().parent;state={'pid':os.getpid(),'completed':[],'protocol':'Recorded-prefix continuation diagnostics, not full resets'}
def status(**kw):
 state.update(kw);p=O/'diagnostics-status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(O/'diagnostics-status.json')
try:
 for variant,name in [('act','native50'),('act-proprio','proprio')]:
  status(phase='waiting_for_export',variant=variant)
  while not (O/'exports'/variant/'export.json').exists():time.sleep(5)
  status(phase='diagnostic',variant=variant)
  with (O/f'diagnostic-{name}.log').open('w') as f:
   p=subprocess.Popen([sys.executable,'-u',str(O/'teacher_diagnostic.py'),'--weights',str(O/'exports'/variant),'--out',str(O/f'diagnostic-{name}')],cwd=O,stdout=f,stderr=subprocess.STDOUT);status(child_pid=p.pid);assert p.wait()==0,f'{variant} diagnostic failed'
  s=json.loads((O/f'diagnostic-{name}/summary.json').read_text());assert s['n']==8 and not s['errors'];state['completed'].append({'variant':variant,'successes':s['successes'],'n':8})
 status(phase='completed')
except Exception as e:status(phase='failed',error=repr(e));raise

from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback
O=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--kind',required=True);a=p.parse_args()
run=O/('run-'+a.kind);receipt=O/('pipeline-'+a.kind+'.json')
def status(stage,**kw):receipt.write_text(json.dumps({'kind':a.kind,'pid':os.getpid(),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'stage':stage,'time':time.time(),**kw},indent=2))
try:
    status('training')
    subprocess.run([sys.executable,str(O/'train.py'),'--kind',a.kind,'--out',str(run),'--steps','12000','--seconds','1800'],check=True)
    status('evaluating')
    subprocess.run([sys.executable,str(O/'evaluate.py'),'--checkpoint',str(run/'best.pt'),'--out',str(run/'fresh-evaluation'),'--device','cuda'],check=True)
    status('completed',summary=json.loads((run/'fresh-evaluation/summary.json').read_text()))
except Exception:
    status('failed',error=traceback.format_exc());raise

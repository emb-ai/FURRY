"""Two bounded, predeclared arms, both initialized from the original policy."""
import argparse,json,signal,subprocess,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);child=None;stopping=False;results={}
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
        if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    common=['--assets','assets','--dataset','dataset','--initial','initial_actor.pt','--source','upstream/actor_critic_future.py',
            '--envs','32','--horizon','256','--minibatch-size','2048','--iterations','1500','--eval-every','100',
            '--eval-seeds','0','1','--actor-lr','3e-6','--actor-warmup-updates','100','--target-kl','.008',
            '--critic-lr','1e-4','--critic-max-updates','5000','--critic-workers','4','--critic-probe-episodes','12',
            '--train-eval-clips','8','--critic-patience','3','--regression-patience','3','--max-hours','3.4']
    arms=[('old_slip',['--legacy-slip']),('new_slip_w2',['--slide-weight','2'])]
    plan={'arms':arms,'common_arguments':common,'wall_time_per_arm_hours':3.4,'baseline':'initial_actor.pt',
          'selection':'validation improvement, no train-subset/replay regression, healthy complete-rollout critic; never deploy automatically',
          'dataset_manifest':json.loads(Path('dataset/manifest.json').read_text())}
    (a.output/'experiment.json').write_text(json.dumps(plan,indent=2))
    for name,flags in arms:
        if stopping:break
        output=a.output/name;cmd=[sys.executable,'training/train.py',*common,'--output',str(output),*flags]
        print(json.dumps({'phase':'night_arm_start','arm':name,'command':cmd}),flush=True)
        with (a.output/(name+'.log')).open('w',buffering=1) as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);code=child.wait()
        result=json.loads((output/'result.json').read_text()) if (output/'result.json').exists() else {'stop_reason':'missing_result'}
        results[name]={'exit_code':code,**result};(a.output/'night-results.json').write_text(json.dumps(results,indent=2))
        print(json.dumps({'phase':'night_arm_complete','arm':name,**results[name]}),flush=True)
        if code:raise RuntimeError(f'{name} failed with exit {code}; see its log')
        if stopping:break
    print(json.dumps({'phase':'night_complete','stopped':stopping,'results':results}),flush=True)

if __name__=='__main__':main()

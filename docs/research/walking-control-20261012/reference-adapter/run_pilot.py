"""One finite smoke -> 300-update pilot -> paired validation pipeline."""
import argparse,json,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--base-source',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=300)
    a,rest=p.parse_known_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    script=str(Path(__file__).with_name('train_adapter.py'))
    common=[sys.executable,script,'--root',a.root,'--base-source',a.base_source]
    def state(phase,**extra):
        target=out/'status.json';tmp=out/'status.tmp';tmp.write_text(json.dumps(dict(phase=phase,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra),indent=2));tmp.replace(target)
    def run(label,flags,cli=rest):
        state(label);argv=common+['--output',str(out/label)]+flags+cli
        (out/(label+'.argv.json')).write_text(json.dumps(argv,indent=2))
        with (out/(label+'.log')).open('w') as log:
            code=subprocess.call(argv,stdout=log,stderr=subprocess.STDOUT)
        if code:state('failed',stage=label,returncode=code);raise RuntimeError(label+' failed')
    run('smoke',['--smoke'])
    eval_cli=list(rest);idx=eval_cli.index('--num_envs');eval_cli[idx+1]='72'
    run('preflight_identity',['--identity','--eval-steps','50'],eval_cli)
    run('preflight_candidate',['--evaluate',str(out/'smoke/adapter_000002.pt'),'--eval-steps','50'],eval_cli)
    pre_a=json.loads((out/'preflight_identity/result.json').read_text())
    pre_b=json.loads((out/'preflight_candidate/result.json').read_text())
    if pre_a['initial_state_sha256']!=pre_b['initial_state_sha256']:
        state('failed',stage='preflight',reason='Initial state mismatch');raise RuntimeError('Unpaired preflight')
    run('training',['--iterations',str(a.iterations)])
    run('identity',['--identity'],eval_cli)
    checkpoint=out/'training'/('adapter_%06d.pt'%a.iterations)
    run('candidate',['--evaluate',str(checkpoint)],eval_cli)
    baseline=json.loads((out/'identity/result.json').read_text());candidate=json.loads((out/'candidate/result.json').read_text())
    if baseline['initial_state_sha256']!=candidate['initial_state_sha256']:
        state('failed',stage='comparison',reason='Initial state mismatch');raise RuntimeError('Unpaired validation')
    import numpy as np
    mode=np.array(json.loads((out/'identity/initial-state.json').read_text())['mode'])
    summary={}
    for m,name in enumerate(['recorded','held_position','held_turn']):
        select=mode==m;row={}
        for key in ['final_distance_m','final_yaw_error_rad','durations_s']:
            before=np.array(baseline[key])[select];after=np.array(candidate[key])[select]
            row[key]=dict(identity_mean=float(before.mean()),candidate_mean=float(after.mean()),mean_delta=float((after-before).mean()))
        b=np.array(baseline['per_env_failure'])[select];c=np.array(candidate['per_env_failure'])[select]
        row['failures']=dict(identity=int(b.sum()),candidate=int(c.sum()),new_failed_trials=int((c&~b).sum()))
        for key in ['low_moving_cost','contact_slip_cost']:
            row[key]=dict(identity=baseline['mean'][key][m],candidate=candidate['mean'][key][m])
        summary[name]=row
    result=dict(completed=True,iterations=a.iterations,paired_initial_state=True,results=summary,
                checkpoint=str(checkpoint),headset_deployment=False,accepted_for_deployment=False,
                caveat='One training seed. Same previously inspected Quest validation session. This experiment is not a reproduction of ReActor.')
    (out/'result.json').write_text(json.dumps(result,indent=2));state('complete',result=str(out/'result.json'))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

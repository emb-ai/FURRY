"""Finite smoke, paired preflight, training blocks and periodic paired validation."""
import argparse,json,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--base-source',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=300)
    p.add_argument('--resume',type=Path);p.add_argument('--eval-every',type=int,default=500)
    a,rest=p.parse_known_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    if a.iterations<=0 or a.eval_every<=0:raise ValueError('Iteration counts must be positive')
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
    resume_flags=['--resume',str(a.resume)] if a.resume else []
    run('smoke',['--smoke']+resume_flags)
    smoke_result=json.loads((out/'smoke/result.json').read_text())
    smoke_checkpoint=out/'smoke'/('adapter_%06d.pt'%smoke_result['final_iteration'])
    eval_cli=list(rest);idx=eval_cli.index('--num_envs');eval_cli[idx+1]='72'
    run('preflight_identity',['--identity','--eval-steps','50'],eval_cli)
    run('preflight_candidate',['--evaluate',str(smoke_checkpoint),'--eval-steps','50'],eval_cli)
    pre_a=json.loads((out/'preflight_identity/result.json').read_text())
    pre_b=json.loads((out/'preflight_candidate/result.json').read_text())
    if pre_a['initial_state_sha256']!=pre_b['initial_state_sha256']:
        state('failed',stage='preflight',reason='Initial state mismatch');raise RuntimeError('Unpaired preflight')
    run('identity',['--identity'],eval_cli)
    if a.resume:
        run('starting_candidate',['--evaluate',str(a.resume)],eval_cli)
        (out/'starting-comparison.json').write_text(json.dumps(
            compare(out,'starting_candidate',a.resume,smoke_result['start_iteration']),indent=2))
    remaining=a.iterations;current=a.resume;comparisons=[]
    while remaining>0:
        count=min(remaining,a.eval_every)
        stage='training' if current is None else 'training_block_%02d'%(len(comparisons)+1)
        run(stage,['--iterations',str(count)]+(['--resume',str(current)] if current else []))
        result=json.loads((out/stage/'result.json').read_text())
        current=out/stage/('adapter_%06d.pt'%result['final_iteration'])
        label='candidate_%06d'%result['final_iteration']
        run(label,['--evaluate',str(current)],eval_cli)
        comparison=compare(out,label,current,result['final_iteration'])
        comparisons.append(comparison);remaining-=count
        (out/'comparisons.json').write_text(json.dumps(comparisons,indent=2))
    result=dict(completed=True,additional_iterations=a.iterations,comparisons=comparisons,
                headset_deployment=False,accepted_for_deployment=False)
    (out/'result.json').write_text(json.dumps(result,indent=2));state('complete',result=str(out/'result.json'))
    print(json.dumps(result),flush=True)


def compare(out,label,checkpoint,iteration):
    baseline=json.loads((out/'identity/result.json').read_text());candidate=json.loads((out/label/'result.json').read_text())
    if baseline['initial_state_sha256']!=candidate['initial_state_sha256']:
        raise RuntimeError('Unpaired validation')
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
    result=dict(completed=True,iterations=iteration,paired_initial_state=True,results=summary,
                checkpoint=str(checkpoint),headset_deployment=False,accepted_for_deployment=False,
                caveat='One training seed. Same previously inspected Quest validation session. This experiment is not a reproduction of ReActor.')
    return result


if __name__=='__main__':main()

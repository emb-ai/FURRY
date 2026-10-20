"""BC initialization + PPO with physical evaluation-gated recovery/sequence curriculum."""
import argparse, json, os, time, hashlib, traceback
from pathlib import Path
import numpy as np
import torch
from curriculum_env import CurriculumEnv, STAGES, BOUNDS, VALID_MACROS, ROOT
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


def atomic_json(path, value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

class Progress(BaseCallback):
    def __init__(self,run,stage):
        super().__init__();self.run=run;self.stage=stage;self.episodes=[];self.moves=0;self.started=time.monotonic()
    def _on_step(self):
        for info,done in zip(self.locals['infos'],self.locals['dones']):
            self.moves+=int(info.get('completed_this_step',False))
            if done:self.episodes.append({k:info[k] for k in ['success','failure','completed_moves','completed_logical_moves','elapsed_seconds','warm_start']})
        if self.n_calls%64==0:
            atomic_json(self.run/'live.json',dict(status='training',stage=self.stage,stage_name=STAGES[self.stage]['name'],
                optimizer_timesteps=int(self.model.num_timesteps),ppo_update_epochs=int(self.model._n_updates),block_completed_quarters=self.moves,
                block_episodes=len(self.episodes),block_successes=sum(x['success'] for x in self.episodes),
                wall_seconds=time.monotonic()-self.started,updated_unix=time.time(),pid=os.getpid()))
        return not (self.run/'STOP').exists()


def factory(seed):
    def make():return Monitor(CurriculumEnv(seed=seed))
    return make


def bc_initialize(model,stage,steps,run):
    data=np.load(ROOT/'demonstrations.npz')
    macro=data['macro_ids']; frame=data['source_indices']
    if stage<4:
        seconds=[1.,2.,5.,24.][stage]
        end=np.array([BOUNDS[min(int(m)+1,9)]*100 for m in macro])
        selected=(macro<9)&(frame>=end-seconds*100)
    else:selected=(macro<9) if stage==4 else np.ones(len(macro),dtype=bool)
    train=np.flatnonzero(selected&data['train']);valid=np.flatnonzero(selected&~data['train'])
    assert len(train) and len(valid)
    obs=torch.from_numpy(data['observations']);act=torch.from_numpy(data['actions'])
    # Wrist deltas must not disappear behind scalar Grip's much larger variance.
    scale=np.maximum(np.std(data['actions'][train],axis=0),np.r_[np.full(12,.015),.1,.1])
    scale=torch.tensor(scale,dtype=torch.float32)
    optimizer=torch.optim.Adam(list(model.policy.mlp_extractor.policy_net.parameters())+list(model.policy.action_net.parameters()),lr=1e-4)
    rng=np.random.default_rng(2000+stage);records=[]
    for i in range(steps):
        ids=rng.choice(train,128)
        mean=model.policy.get_distribution(obs[ids]).distribution.mean
        loss=((mean-act[ids])/scale).square().mean()
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.policy.parameters(),1.);optimizer.step()
        if i==steps-1 or i%250==0:
            with torch.no_grad():
                val=model.policy.get_distribution(obs[valid]).distribution.mean
                val_loss=float(((val-act[valid])/scale).square().mean())
            records.append(dict(update=i+1,train=float(loss.detach()),validation=val_loss))
    # Small but nonzero physical exploration: ~0.9 mm wrist / 0.26 deg / .015 Grip.
    with torch.no_grad():model.policy.log_std.fill_(-3.5)
    atomic_json(run/f'stage-{stage}-bc.json',dict(stage=stage,train_samples=len(train),validation_samples=len(valid),
        weighted_loss=True,scales=scale.tolist(),records=records,physical_success_not_implied=True))
    model.save(run/f'stage-{stage}-bc')


def evaluate(model,env,stage,run,index):
    env.set_stage(stage)
    cases=[]
    if stage<4:
        for holdout,macros in [(False,(0,4)),(True,VALID_MACROS)]:
            for macro in macros:
                for offset in STAGES[stage]['offsets']:
                    bi=next(i for i,e in enumerate(env.bank) if e['macro']==macro and e['offset']==offset)
                    cases.append((holdout,dict(bank_index=bi),10000+macro))
    else:
        cases=[(True,{},10000+i) for i in range(8)]
    results=[]
    for case,(holdout,options,seed) in enumerate(cases):
        env.evaluation=holdout
        obs,reset_info=env.reset(seed=seed,options=options)
        initial_metrics=env.geometry.metrics(env.target)
        reward_sum=0;actions=0
        while True:
            action,_=model.predict(obs,deterministic=True)
            obs,reward,done,truncated,info=env.step(action);reward_sum+=reward;actions+=1
            if done or truncated:break
        results.append(dict(holdout=holdout,seed=seed,reset=reset_info,initial_metrics=initial_metrics,
            final=info,reward=reward_sum,actions=actions))
        atomic_json(run/'live.json',dict(status='evaluation',stage=stage,stage_name=STAGES[stage]['name'],
            evaluation=index,finished_cases=case+1,total_cases=len(cases),
            successes=sum(r['final']['success'] for r in results),updated_unix=time.time(),pid=os.getpid()))
        if (run/'STOP').exists():break
    held=[r for r in results if r['holdout']];train=[r for r in results if not r['holdout']]
    rate=lambda xs:sum(r['final']['success'] for r in xs)/len(xs) if xs else None
    report=dict(stage=stage,name=STAGES[stage]['name'],evaluation=index,results=results,
        holdout_success=rate(held),train_success=rate(train),
        independent_physical_start=False,complete=len(results)==len(cases),
        warm_start=stage<4,optimizer_timesteps=int(model.num_timesteps))
    atomic_json(run/f'eval-{stage}-{index:03d}.json',report)
    print('EVAL',json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',default='curriculum-v1')
    ap.add_argument('--envs',type=int,default=2)
    ap.add_argument('--block-steps',type=int,default=4096)
    ap.add_argument('--stage-steps',type=int,default=65536)
    ap.add_argument('--total-steps',type=int,default=500000)
    ap.add_argument('--bc-updates',type=int,default=1500)
    ap.add_argument('--start-stage',type=int,default=0)
    ap.add_argument('--load',default=str(ROOT/'bc-policy.zip'))
    args=ap.parse_args();torch.set_num_threads(2);torch.manual_seed(413)
    run=ROOT/args.run;run.mkdir(exist_ok=True)
    if (run/'status.json').exists():raise RuntimeError('Use a new run directory; do not overwrite a run')
    provenance={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['native_env.py','curriculum_env.py','train_curriculum.py']}
    status=dict(status='starting',pid=os.getpid(),arguments=vars(args),code_sha256=provenance,
        started_unix=time.time(),stages=STAGES,accepted_policy=None,
        actions='two wrists (12 coordinates) + two ordinary scalar Grip',
        cube_in_hands=True,no_support=True,physics_modified=False,lower_policy='frozen HOMIE Standing',
        initialization='recorded native replay snapshots only at episode boundaries',
        device='local CPU',headset_deployed=False)
    atomic_json(run/'status.json',status)
    vec=None;ev=None;total=0;completed=[]
    try:
        vec=SubprocVecEnv([factory(413+i) for i in range(args.envs)],start_method='spawn')
        ev=CurriculumEnv(evaluation=True,seed=900)
        atomic_json(run/'bank.json',ev.bank_report)
        model=PPO.load(args.load,env=vec,device='cpu',learning_rate=1e-5,target_kl=.02,
                       n_steps=256,batch_size=128,n_epochs=4,verbose=0)
        model.num_timesteps=0
        for stage in range(args.start_stage,len(STAGES)):
            vec.env_method('set_stage',stage);vec.reset();model._last_obs=None
            status.update(status='bc_initialization',stage=stage,stage_name=STAGES[stage]['name'],updated_unix=time.time())
            atomic_json(run/'status.json',status)
            bc_initialize(model,stage,args.bc_updates,run)
            # BC changed the actor; stale PPO optimizer momentum must not be reused.
            model.policy.optimizer=torch.optim.Adam(model.policy.parameters(),lr=1e-5,eps=1e-5)
            streak=0;stage_steps=0;index=0
            while True:
                status.update(status='evaluation',optimizer_timesteps=total,completed_stages=completed,updated_unix=time.time())
                atomic_json(run/'status.json',status)
                report=evaluate(model,ev,stage,run,index);index+=1
                passed=(report['complete'] and report['holdout_success']>=.875 and
                        (report['train_success'] is None or report['train_success']>=.875))
                streak=streak+1 if passed else 0
                # Two consecutive checks separated by learning; warm validation
                # is a correlated macro holdout, never called general robustness.
                if streak>=2:
                    completed.append(stage);model.save(run/f'stage-{stage}-passed');break
                if total>=args.total_steps or stage_steps>=args.stage_steps or (run/'STOP').exists():
                    status.update(status='stopped_before_gate',reason='budget_or_stop',stage=stage,
                        optimizer_timesteps=total,completed_stages=completed,last_evaluation=report['holdout_success'],
                        finished_unix=time.time())
                    model.save(run/'final');atomic_json(run/'status.json',status);return
                steps=min(args.block_steps,args.total_steps-total,args.stage_steps-stage_steps)
                status.update(status='training',updated_unix=time.time());atomic_json(run/'status.json',status)
                progress=Progress(run,stage);before=model.num_timesteps
                actor_before=torch.cat([p.detach().flatten().cpu() for p in model.policy.action_net.parameters()]).clone()
                model.learn(total_timesteps=steps,reset_num_timesteps=False,callback=progress)
                actual=int(model.num_timesteps-before);total+=actual;stage_steps+=actual
                atomic_json(run/f'train-{stage}-{index:03d}.json',dict(stage=stage,steps=actual,
                    episodes=progress.episodes,completed_quarters=progress.moves,
                    ppo_update_epochs=int(model._n_updates),
                    actor_action_head_max_change=float((torch.cat([p.detach().flatten().cpu() for p in model.policy.action_net.parameters()])-actor_before).abs().max())))
                model.save(run/'latest')
        status.update(status='curriculum_gates_passed',completed_stages=completed,optimizer_timesteps=total,
            accepted_policy=None,reason='requires separate longer unseen-sequence evaluation',finished_unix=time.time())
        model.save(run/'final');atomic_json(run/'status.json',status)
    except BaseException:
        status.update(status='failed',error=traceback.format_exc(),finished_unix=time.time())
        atomic_json(run/'status.json',status);raise
    finally:
        if vec is not None:vec.close()
        if ev is not None:ev.close()

if __name__=='__main__':main()

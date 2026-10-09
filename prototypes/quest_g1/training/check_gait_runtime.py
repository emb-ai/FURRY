"""Real MuJoCo/ONNX check: diagnostics preserve dynamics, 50 Hz holds actions/history."""
import argparse,json
import numpy as np
import onnxruntime as ort
from env import Env,Motions
from gait_evaluation import runtime_env_class,RAW_COLUMNS,summarize_gait
from rewards import RewardConfig


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--dataset',required=True);p.add_argument('--policy',required=True);p.add_argument('--output');a=p.parse_args()
    motions=Motions(a.dataset,'validation');opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    policy=ort.InferenceSession(a.policy,sess_options=opts,providers=['CPUExecutionProvider']);name=policy.get_inputs()[0].name
    results=[]
    for hz in (100,50):
        base=Env(a.assets,motions,reward_config=RewardConfig(slide_weight=2.))
        actual=runtime_env_class(Env,hz,True)(a.assets,motions,reward_config=RewardConfig(slide_weight=2.))
        expected_obs=base.reset(clip=0,start=0);obs=actual.reset(clip=0,start=0);calls=0;traces=[]
        for tick in range(200):
            if tick%actual.policy_stride==0:
                action=policy.run(None,{name:obs[None]})[0][0];calls+=1
            old_history=actual.hist.copy()
            expected_obs,expected_reward,expected_done,expected_info=base.step(action)
            obs,reward,done,info=actual.step(action)
            assert np.array_equal(base.data.qpos,actual.data.qpos),(hz,tick,'qpos')
            assert np.array_equal(base.data.qvel,actual.data.qvel),(hz,tick,'qvel')
            assert reward==expected_reward and done==expected_done,(hz,tick,'reward/done')
            if hz==100:assert np.array_equal(expected_obs,obs),(hz,tick,'observation')
            elif actual.steps%2:assert np.array_equal(old_history,actual.hist),(hz,tick,'intermediate history update')
            else:assert np.array_equal(obs[:127],expected_obs[:127]),(hz,tick,'current state')
            traces.append([info[k] for k in RAW_COLUMNS])
            if done:break
        assert calls==(len(traces)+actual.policy_stride-1)//actual.policy_stride
        results.append({'control_hz':hz,'physics_steps':len(traces)*10,'policy_calls':calls,'state_reward_and_termination_bitwise_equal':True,'gait':summarize_gait(np.array(traces),RAW_COLUMNS)})
    print(json.dumps(results,indent=2))
    if a.output:
        from pathlib import Path
        Path(a.output).write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()

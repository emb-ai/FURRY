"""Check that full traces preserve states/observations and log consumed inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
from env import Env, Motions
from full_recorder import instrument
from gait_evaluation import runtime_env_class


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assets', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    motions = Motions(a.dataset, 'train')
    Plain = runtime_env_class(Env, 50, True)
    Recorded = runtime_env_class(instrument(Env, a.output), 50, True)
    plain, recorded = Plain(a.assets, motions), Recorded(a.assets, motions)
    plain.max_steps = recorded.max_steps = 20
    recorded.evaluation_seed = 0
    plain.reset(clip=0, start=0); recorded.reset(clip=0, start=0)
    maximum = 0.; expected = []
    for i in range(20):
        expected.append(plain.last_obs.copy())
        action = np.sin(np.arange(29)*.1+(i//2)*.02)*.1
        x, y = plain.step(action), recorded.step(action)
        maximum = max(maximum, float(np.abs(plain.data.qpos-recorded.data.qpos).max()),
                      float(np.abs(x[0]-y[0]).max()))
        if x[2] or y[2]:
            assert x[2] == y[2]
            break
    traces = list(a.output.glob('*.npz')); assert len(traces) == 1
    with np.load(traces[0]) as z:
        assert np.isfinite(z['qpos']).all() and len(z['qpos']) == i+1
        assert z['contact_offsets'][-1] == len(z['contacts']) and z['contacts'].shape[1] == 11
        np.testing.assert_array_equal(z['actor_input'], expected)
        np.testing.assert_array_equal(z['actor_history'], z['actor_input'][:, 127:1397].reshape(-1, 10, 127))
        np.testing.assert_array_equal(z['command'], z['actor_input'][:, :35])
        np.testing.assert_array_equal(z['policy_tick'], np.arange(i+1)%2 == 0)
        np.testing.assert_allclose(z['state_time_s']-z['input_time_s'], .01)
        result = {'steps': i+1, 'state_and_observation_max_error': maximum,
                  'contact_rows': len(z['contacts']), 'consumed_actor_input_matches': True,
                  'passed': maximum == 0., 'scope': 'one deterministic CPU smoke; not policy quality'}
    (a.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result)); assert result['passed']


if __name__ == '__main__':
    main()

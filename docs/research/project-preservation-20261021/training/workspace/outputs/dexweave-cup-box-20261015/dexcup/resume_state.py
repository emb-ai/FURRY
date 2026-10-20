"""Restore saved PPO state with strict parity checks before any update."""
import hashlib
import torch


def equal_state(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual.detach().cpu(), expected.detach().cpu())
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected: equal_state(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected): equal_state(a, e)
    else:
        assert actual == expected


def restore_training_state(path, policy, algorithm, normalizer, critic_normalizer, contract, native):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    saved = checkpoint['contract']
    for key in ['schema','reference_sha256','seed','num_envs','actor_dim','critic_dim','action_dim','joints','links','settings','randomize']:
        assert saved[key] == contract[key], f'Incompatible training contract: {key}'
    for key in ['joint_names','body_names','action_names','default_q','scale','dt','control_dt']:
        assert checkpoint['native_contract'][key] == native[key], f'Incompatible native contract: {key}'
    policy.load_state_dict(checkpoint['policy'], strict=True)
    algorithm.optimizer.load_state_dict(checkpoint['optimizer'])
    normalizer.load_state_dict(checkpoint['normalizer'], strict=True)
    critic_normalizer.load_state_dict(checkpoint['critic_normalizer'], strict=True)
    # PPO keeps adaptive LR separately from the optimizer; restore both.
    rates = {float(group['lr']) for group in algorithm.optimizer.param_groups}
    assert len(rates) == 1
    algorithm.learning_rate = rates.pop()
    for actual, key in [(policy.state_dict(),'policy'), (algorithm.optimizer.state_dict(),'optimizer'),
                        (normalizer.state_dict(),'normalizer'),(critic_normalizer.state_dict(),'critic_normalizer')]:
        equal_state(actual,checkpoint[key])
    return {'checkpoint':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'iteration':checkpoint['iteration'],'policy_exact':True,'optimizer_exact':True,
            'normalizers_exact':True,'learning_rate':algorithm.learning_rate,
            'environment_restarted':True,'sampler_statistics_restarted':True,'rng_restarted':True,
            'bitwise_continuation':False,
            'limitation':'Original checkpoint omits simulator, RNG and adaptive phase-sampling counters.'}

"""Actor-only commands from raw Meta -> native GMR; no reward target mutation.

This is an explicit, mode-tagged input experiment. Recorded commands do not
provide measured human foot phase, nor closed-loop catch-up for a new actor.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NativeCommandBank:
    def __init__(self, path, files, dataset, device):
        path, dataset = Path(path), Path(dataset).resolve()
        self.metadata = json.loads(path.with_suffix('.json').read_text())
        m = self.metadata
        if m.get('schema') != 'quest-native-command-bank-v1':
            raise ValueError('Unknown native command bank schema')
        if m['npz_sha256'] != sha(path) or m['dataset_manifest_sha256'] != sha(dataset/'manifest.json'):
            raise ValueError('Native command bank or dataset changed')
        manifest = json.loads((dataset/'manifest.json').read_text())
        train = {c['file']: c for c in manifest['clips'] if c['split'] == 'train'}
        names = [str(Path(f).resolve().relative_to(dataset)) for f in files]
        if len(set(names)) != len(names) or set(names) != set(train):
            raise ValueError('Loader must contain exactly the pinned TRAIN motions')
        quest = {n for n, c in train.items() if c['source'] == 'quest'}
        if set(m['files']) != quest:
            raise ValueError('Bank must cover every Quest TRAIN motion, and no others')
        # Pack sorted times onto disjoint timelines. One searchsorted per batch;
        # no [4096,max_clip_frames] copies and no interpolation into the future.
        offsets, starts, ends, times, commands, active = [], [], [], [], [], []
        cursor, timeline = 0, 0.
        with np.load(path, allow_pickle=False) as z:
            for name in names:
                active.append(name in quest)
                if name in quest:
                    row = m['files'][name]
                    if row['target_sha256'] != train[name]['sha256']:
                        raise ValueError('Native commands paired with a different reward target')
                    arr = z[row['key']]
                    if arr.ndim != 2 or arr.shape[1] != 36 or not np.isfinite(arr).all():
                        raise ValueError('Invalid command timeline')
                    if len(arr) < 2 or arr[0, 0] != 0 or not (np.diff(arr[:, 0]) > 0).all():
                        raise ValueError('Native command times must start at zero and increase')
                else:
                    arr = np.zeros((1, 36), np.float64)
                offsets.append(timeline); starts.append(cursor); ends.append(cursor+len(arr)-1)
                times.append(arr[:, 0]+timeline); commands.append(arr[:, 1:])
                cursor += len(arr)
                timeline += max(float(train[name]['seconds']), float(arr[-1, 0])) + 10.
        self.times = torch.tensor(np.concatenate(times), dtype=torch.float64, device=device)
        self.commands = torch.tensor(np.concatenate(commands), dtype=torch.float32, device=device)
        self.offsets = torch.tensor(offsets, dtype=torch.float64, device=device)
        self.starts = torch.tensor(starts, dtype=torch.long, device=device)
        self.ends = torch.tensor(ends, dtype=torch.long, device=device)
        self.active = torch.tensor(active, dtype=torch.bool, device=device)
        self.lengths = torch.tensor([train[n]['seconds'] for n in names], device=device)

    def sample(self, ids, times, fallback):
        query = self.offsets[ids] + times.double().clamp(min=0)
        index = torch.searchsorted(self.times, query, right=True)-1
        index = torch.maximum(self.starts[ids], torch.minimum(index, self.ends[ids]))
        selected = self.commands[index].clone()
        beyond = times > self.lengths[ids]
        selected[:, (0, 1, 5)] *= (~beyond)[:, None]
        return torch.where(self.active[ids, None], selected, fallback)


class QuestActorHistory:
    """Build the native 1432-vector, oldest history first, zero after reset."""
    def __init__(self, count, device):
        self.history = torch.zeros(count, 10, 127, device=device)

    def reset(self, ids):
        self.history[ids] = 0

    def observe(self, current):
        if current.shape != (len(self.history), 127):
            raise ValueError('Unexpected Quest current-observation contract')
        out = torch.cat((current, self.history.flatten(1), current[:, :35]), dim=-1)
        self.history[:, :-1] = self.history[:, 1:].clone()
        self.history[:, -1] = current
        return out


def install(env, path, dataset):
    if abs(env.dt-.02) > 1e-8 or env.num_obs != 1432 or env.obs_type != 'student_future':
        raise ValueError('Expected pinned 50 Hz / 1432-observation student actor')
    bank = NativeCommandBank(path, env._motion_lib._motion_files, dataset, env.device)
    history = QuestActorHistory(env.num_envs, env.device)
    original_obs, original_reset = env.compute_observations, env.reset_idx
    counters = torch.zeros(2, dtype=torch.long, device=env.device)

    def observe():
        # Source computes reward references/critic/proprio exactly as before.
        # Overwrite only actor command and actor history AFTER source assembly.
        original_obs()
        current = env.obs_buf[:, :127].clone()
        current[:, :35] = bank.sample(env._motion_ids, env._get_motion_times(), current[:, :35])
        env.obs_buf = history.observe(current)
        env.obs_history_buf.copy_(history.history)
        counters[0] += bank.active[env._motion_ids].sum()
        counters[1] += env.num_envs

    def reset(ids, *args, **kwargs):
        original_reset(ids, *args, **kwargs)
        history.reset(ids)

    env.compute_observations, env.reset_idx = observe, reset
    # Environment constructed before installation; rebuild its initial actor
    # observation without retroactively filling history with unavailable frames.
    # Before the first physics step the source has no roll/pitch and leaves the
    # actor buffer zero, as on the legacy path; keep it.
    if hasattr(env, 'roll'):
        observe()
    return {'bank': bank, 'history': history, 'counters': counters}

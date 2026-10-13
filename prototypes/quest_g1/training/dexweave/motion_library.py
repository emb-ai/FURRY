"""Validated training-only motion bank and clip-local phase sampling."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

KEYS = ('joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w')

def load_bank(manifest, joint_names, body_names):
    path = Path(manifest)
    data = json.loads(path.read_text())
    if data.get('schema') != 'furry-dexweave-multiclip-v1' or data.get('role') != 'train' or data.get('fps') != 50:
        raise ValueError('Expected a 50 Hz training manifest')
    clips = data['clips']
    if not clips or len({c['id'] for c in clips}) != len(clips) or len({c['sha256'] for c in clips}) != len(clips):
        raise ValueError('Empty or duplicate training clips')
    arrays = {k: [] for k in KEYS}; lengths = []
    for clip in clips:
        if clip.get('split') != 'train':
            raise ValueError('Non-training clip in training manifest')
        source = (path.parent / clip['file']).resolve()
        if not source.is_relative_to(path.parent.resolve()):
            raise ValueError('Clip outside manifest directory')
        if hashlib.sha256(source.read_bytes()).hexdigest() != clip['sha256']:
            raise ValueError('Clip SHA256 mismatch')
        with np.load(source, allow_pickle=False) as f:
            n = len(f['joint_pos'])
            if int(f['fps']) != 50 or n != clip['frames'] or n < 4:
                raise ValueError('Invalid clip frame count or frequency')
            ji = [f['joint_names'].tolist().index(x) for x in joint_names]
            bi = [f['body_names'].tolist().index(x) for x in body_names]
            for key in KEYS:
                a = f[key][:, ji if key.startswith('joint') else bi]
                if len(a) != n or not np.isfinite(a).all():
                    raise ValueError('Invalid motion array')
                arrays[key].append(a)
            lengths.append(n)
    return {k: np.concatenate(v) for k, v in arrays.items()}, lengths, clips

class ClipSampler:
    """Fixed balanced environment quotas; every sampled phase stays in its clip.

    Equal phase bins (~1 second, at least four) make the .28 cap feasible
    even for the short 2.34-second trajectory. Four water-filling rounds
    suffice: no more than three bins can saturate at probability .28.
    """
    def __init__(self, lengths, num_envs, device):
        self.lengths = torch.tensor(lengths, device=device, dtype=torch.long)
        self.starts = self.lengths.cumsum(0) - self.lengths
        self.clip_ids = torch.arange(num_envs, device=device) % len(lengths)
        self.ends = (self.starts + self.lengths - 1)[self.clip_ids]
        self.bin_counts = ((self.lengths + 49) // 50).clamp(min=4)
        self.max_bins = int(self.bin_counts.max())
        self.valid = torch.arange(self.max_bins, device=device)[None] < self.bin_counts[:, None]
        self.visits = torch.zeros_like(self.valid, dtype=torch.float32)
        self.failures = torch.zeros_like(self.visits)
        self.resets = torch.zeros(len(lengths), device=device, dtype=torch.long)

    def probabilities(self):
        w = (torch.sqrt(self.failures / self.visits.clamp(min=1)) + 1e-8) * self.valid
        fixed = torch.zeros_like(self.valid)
        result = torch.zeros_like(w)
        for _ in range(4):
            remaining = w * ~fixed
            p = remaining / remaining.sum(1, keepdim=True).clamp(min=1e-30) * (1 - result.sum(1, keepdim=True))
            newly = (p > .28) & ~fixed & self.valid
            result = torch.where(newly, .28, result)
            fixed |= newly
        remaining = w * ~fixed
        return result + remaining / remaining.sum(1, keepdim=True).clamp(min=1e-30) * (1 - result.sum(1, keepdim=True))

    def sample(self, ids, steps, failed):
        clip = self.clip_ids[ids]; lengths = self.lengths[clip]
        local = (steps[ids] - self.starts[clip]).clamp(min=0)
        bins = torch.minimum(local * self.bin_counts[clip] // lengths, self.bin_counts[clip] - 1)
        flat = clip * self.max_bins + bins
        self.visits += torch.bincount(flat, minlength=self.visits.numel()).reshape_as(self.visits)
        self.failures += torch.bincount(flat[failed], minlength=self.visits.numel()).reshape_as(self.visits)
        self.resets += torch.bincount(clip, minlength=len(self.lengths))
        # Draw from the per-clip distribution; no global concatenation seams.
        chosen = torch.multinomial(self.probabilities()[clip], 1).squeeze(1)
        low = (chosen * lengths + self.bin_counts[clip] - 1) // self.bin_counts[clip]
        high = ((chosen + 1) * lengths + self.bin_counts[clip] - 1) // self.bin_counts[clip]
        rand = lambda: torch.rand(len(ids), device=ids.device)
        adaptive = low + (rand() * (high - low)).long()
        uniform = (rand() * lengths).long()
        phase = torch.where(rand() < .3, adaptive, uniform)
        sampled = rand() >= .5
        phase = (phase - torch.randint(25, 76, (len(ids),), device=ids.device)).clamp(min=0)
        return self.starts[clip] + torch.where(sampled, phase, 0), sampled

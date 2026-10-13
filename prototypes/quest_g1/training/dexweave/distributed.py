"""Global training-only observation statistics and replica checks."""
import hashlib
import torch
import torch.distributed as dist
from rsl_rl.modules import EmpiricalNormalization

class GlobalNormalization(EmpiricalNormalization):
    @torch.no_grad()
    def update(self, x):
        if self.until is not None and self.count >= self.until:
            return
        # Float64 avoids cancellation when combining batch second moments.
        v = x.double()
        packed = torch.cat([v.sum(0), v.square().sum(0), v.new_tensor([len(v)])])
        dist.all_reduce(packed)
        d = x.shape[-1]; count = packed[-1].long()
        mean = (packed[:d] / count).to(x.dtype)[None]
        var = (packed[d:2*d] / count - (packed[:d] / count).square()).clamp(min=0).to(x.dtype)[None]
        self.count += count
        rate = count / self.count
        delta = mean - self._mean
        self._mean += rate * delta
        self._var += rate * (var - self._var + delta * (mean - self._mean))
        self._std = self._var.clamp(min=0).sqrt()

@torch.no_grad()
def synchronize_policy(policy):
    for tensor in policy.state_dict().values():
        dist.broadcast(tensor, src=0)

@torch.no_grad()
def replica_hashes(*modules):
    digest = hashlib.sha256()
    for module in modules:
        for key, value in sorted(module.state_dict().items()):
            digest.update(key.encode());digest.update(value.detach().cpu().numpy().tobytes())
    device = next(modules[0].parameters()).device
    own = torch.tensor(list(digest.digest()), device=device, dtype=torch.uint8)
    gathered = [torch.empty_like(own) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, own)
    hashes = [bytes(v.cpu().tolist()).hex() for v in gathered]
    if len(set(hashes)) != 1:
        raise RuntimeError('Policy or normalizer replicas diverged')
    return hashes

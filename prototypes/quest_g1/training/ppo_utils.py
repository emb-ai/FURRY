"""Time-aware PPO arithmetic, isolated for boundary and schedule tests."""
import numpy as np
GAMMA=.99**.5
GAE_LAMBDA=.95**.5

def advantages(rewards, values, dones, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
    # Artificial timeouts have one terminal-state bootstrap added by the collector.
    # A done always stops GAE from crossing into the reset episode.
    out=np.zeros_like(rewards);gae=np.zeros_like(last_value)
    for t in reversed(range(len(rewards))):
        nv=last_value if t==len(rewards)-1 else values[t+1]
        live=1-np.asarray(dones[t],float)
        delta=rewards[t]+gamma*nv*live-values[t]
        gae=delta+gamma*lam*live*gae;out[t]=gae
    return out,out+values

def actor_lr(update, target, warmup_updates):
    if update<1:return 0.
    if warmup_updates<=1:return target
    return target*min(1.,.1+.9*(update-1)/max(1,warmup_updates-1))

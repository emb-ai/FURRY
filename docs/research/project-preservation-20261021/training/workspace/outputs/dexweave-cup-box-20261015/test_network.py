import torch
from dexcup.network import Actor

def test_directed_dependencies():
    torch.manual_seed(4);a=Actor();x=torch.randn(3,244);y=a(x)
    assert y.shape==(3,43) and torch.isfinite(y).all()
    changed=x.clone();changed[:,-27:]+=2
    assert torch.equal(a(changed)[:,:12],y[:,:12])
    assert not torch.equal(a(changed)[:,12:],y[:,12:])
    changed=x.clone()
    for off in (0,102,174):changed[:,off+29:off+43]+=2
    assert torch.equal(a(changed)[:,:29],y[:,:29])
    assert not torch.equal(a(changed)[:,29:],y[:,29:])
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.parameters())

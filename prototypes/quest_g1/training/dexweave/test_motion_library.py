import hashlib
import json
import numpy as np
import pytest
import torch
from .motion_library import ClipSampler, load_bank, KEYS
from .sampling import capped_probabilities


def test_short_and_long_clips_remain_separate_under_resets_and_advances():
    lengths=[118,130,398,1500]
    s=ClipSampler(lengths,4096,'cpu');ids=torch.arange(4096)
    steps=s.starts[s.clip_ids].clone()
    for _ in range(30):
        steps,sampled=s.sample(ids,steps,torch.rand(4096)<.5)
        local=steps-s.starts[s.clip_ids]
        assert (local>=0).all() and (local<s.lengths[s.clip_ids]).all()
        advanced=torch.minimum(steps+10000,s.ends)
        assert torch.equal(advanced,s.ends)
    assert s.resets.tolist()==[30720]*4
    assert torch.bincount(s.clip_ids).tolist()==[1024]*4


def test_batched_cap_matches_single_clip_oracle():
    s=ClipSampler([118,398,1500],30,'cpu')
    torch.manual_seed(49)
    for _ in range(25):
        s.failures[:]=10**(-8*torch.rand_like(s.failures))
        s.visits[:]=1
        p=s.probabilities()
        assert torch.allclose(p.sum(1),torch.ones(3),atol=2e-7)
        assert p.max()<=.280001 and (p[~s.valid]==0).all()
        for i,n in enumerate(s.bin_counts.tolist()):
            expected=capped_probabilities(torch.sqrt(s.failures[i,:n])+1e-8)
            torch.testing.assert_close(p[i,:n],expected,atol=2e-7,rtol=1e-5)


def fixture_manifest(tmp_path):
    arrays={key:np.ones((5,2,3)) for key in KEYS}
    arrays.update(joint_names=['a','b'],body_names=['c','d'],fps=50)
    np.savez(tmp_path/'clip.npz',**arrays)
    clip={'id':'train','file':'clip.npz','sha256':hashlib.sha256((tmp_path/'clip.npz').read_bytes()).hexdigest(),'frames':5,'split':'train'}
    data={'schema':'furry-dexweave-multiclip-v1','fps':50,'role':'train','clips':[clip]}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(data))
    return path,data


def test_loader_loads_only_manifest_members(tmp_path):
    path,data=fixture_manifest(tmp_path)
    (tmp_path/'excluded-test.npz').write_bytes(b'not an NPZ: must never be loaded')
    arrays,lengths,clips=load_bank(path,['b','a'],['d','c'])
    assert lengths==[5] and len(clips)==1
    assert all(len(v)==5 for v in arrays.values())


@pytest.mark.parametrize('bad',['role','split','hash','duplicate'])
def test_loader_rejects_invalid_membership(tmp_path,bad):
    path,data=fixture_manifest(tmp_path)
    if bad=='role':data['role']='test'
    elif bad=='split':data['clips'][0]['split']='test'
    elif bad=='hash':data['clips'][0]['sha256']='0'*64
    else:data['clips']*=2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):load_bank(path,['a','b'],['c','d'])

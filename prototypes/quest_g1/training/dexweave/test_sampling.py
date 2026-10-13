import numpy as np
import pytest
import torch
from .sampling import capped_probabilities


def oracle(weights, cap=.28):
    w=np.maximum(weights,1e-8)
    low,high=0.,1./w.min()
    for _ in range(100):
        mid=(low+high)/2
        if np.minimum(w*mid,cap).sum()>1:high=mid
        else:low=mid
    return np.minimum(w*(low+high)/2,cap)


@pytest.mark.parametrize('bins',[4,10,18,30])
def test_cap_matches_independent_bisection_for_concentrated_failures(bins):
    rng=np.random.default_rng(9)
    cases=[np.zeros(bins),np.ones(bins),np.r_[1.,1.,np.full(bins-2,1e-8)],np.r_[1.,np.zeros(bins-1)]]
    cases += [10.**rng.uniform(-8,0,bins) for _ in range(30)]
    for weights in cases:
        result=capped_probabilities(torch.tensor(weights,dtype=torch.float32)).numpy()
        assert np.isfinite(result).all() and result.min()>=0 and result.max()<=.280001
        np.testing.assert_allclose(result.sum(),1.,atol=2e-7)
        np.testing.assert_allclose(result,oracle(weights),atol=2e-7)


def test_infeasible_cap_is_rejected():
    with pytest.raises(ValueError,match='Too few'):
        capped_probabilities(torch.ones(3))

from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from rsl_rl.modules import EmpiricalNormalization
from .distributed import GlobalNormalization, synchronize_policy, replica_hashes


def worker(rank, path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    norm=GlobalNormalization([3],until=100)
    oracle=EmpiricalNormalization([3],until=100)
    # Deliberately different distributions and counts on each rank.
    for step in range(4):
        all_x=[torch.arange((r+2)*3,dtype=torch.float32).reshape(-1,3)+100*r+step for r in range(2)]
        norm(all_x[rank]);oracle(torch.cat(all_x))
        for key,value in norm.state_dict().items():
            torch.testing.assert_close(value,oracle.state_dict()[key],rtol=1e-5,atol=1e-5)
    model=torch.nn.Linear(3,2)
    with torch.no_grad():model.weight.fill_(rank+1);model.bias.fill_(rank)
    synchronize_policy(model)
    assert torch.equal(model.weight,torch.ones_like(model.weight))
    assert len(set(replica_hashes(model,norm)))==1
    dist.destroy_process_group()


def test_global_statistics_match_concatenated_training_samples(tmp_path):
    mp.spawn(worker,args=(str(tmp_path/'rendezvous'),),nprocs=2,join=True)

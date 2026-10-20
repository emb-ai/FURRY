"""Official LeRobot ACT architecture/optimizer with a FURRY HDF5 adapter.

State input is previous HOMIE command, NOT measured robot proprioception.
Validation episodes stay held out. Loss is not autonomous task success.
"""
import argparse
import json
import random
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


def statistics(path):
    with h5py.File(path, "r") as h:
        names = [x.decode() for x in h["mask/train"][:]]
        result = {}
        for key in ["actions", "obs/previous_command"]:
            x = np.concatenate([h[f"data/{n}/{key}"][:] for n in names])
            result[key] = {"mean": x.mean(0), "std": np.maximum(x.std(0), 1e-4)}
        return result


class CupDataset(Dataset):
    def __init__(self, path, split, stats, chunk=64):
        self.path, self.stats, self.chunk = str(path), stats, chunk
        self.handle = None
        self.index, self.episodes = [], {}
        with h5py.File(path, "r") as h:
            self.names = [x.decode() for x in h[f"mask/{split}"][:]]
            for name in self.names:
                d = h[f"data/{name}"]
                self.episodes[name] = (d["actions"][:], d["obs/previous_command"][:])
                self.index.extend((name, t) for t in range(len(d["actions"])))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        if self.handle is None:
            self.handle = h5py.File(self.path, "r")
        name, t = self.index[i]
        actions, commands = self.episodes[name]
        indices = np.arange(t, t + self.chunk)
        image = self.handle[f"data/{name}/obs/agentview_image"][t].transpose(2, 0, 1).astype(np.float32) / 255
        image = (image - np.array([.485, .456, .406], dtype=np.float32)[:, None, None]) / np.array([.229, .224, .225], dtype=np.float32)[:, None, None]
        a, s = self.stats["actions"], self.stats["obs/previous_command"]
        return {"observation.images.ego": torch.from_numpy(image),
                "observation.state": torch.from_numpy((commands[t] - s["mean"]) / s["std"]),
                "action": torch.from_numpy((actions[np.minimum(indices, len(actions)-1)] - a["mean"]) / a["std"]),
                "action_is_pad": torch.from_numpy(indices >= len(actions))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=5000)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    stats = statistics(args.data)
    train, valid = [CupDataset(args.data, s, stats) for s in ["train", "valid"]]
    assert not set(train.names) & set(valid.names)
    config = ACTConfig(input_features={"observation.images.ego": PolicyFeature(FeatureType.VISUAL, (3,224,224)), "observation.state": PolicyFeature(FeatureType.STATE, (39,))}, output_features={"action": PolicyFeature(FeatureType.ACTION, (39,))}, chunk_size=64, n_action_steps=8, device=args.device)
    policy = ACTPolicy(config).to(args.device)
    optim_config = config.get_optimizer_preset()
    optimizer = optim_config.build(policy.get_optim_params())
    metadata = {"upstream_commit": "53feaef37b67e8f4c23b6953e880ff535e39d6fe", "state_semantics": "previous 39-dimensional HOMIE command, not measured state", "normalization": "train-only mean/std; ImageNet RGB", "backbone": "official ResNet18 ImageNet weights", "train_episodes": train.names, "valid_episodes": valid.names, "autonomous_evaluation": False, "args": vars(args) | {"data":str(args.data),"out":str(args.out)}}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    config.save_pretrained(args.out / "config")
    loader = DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,persistent_workers=args.workers>0,pin_memory=args.device.startswith("cuda"))
    val = DataLoader(valid,batch_size=args.batch_size,num_workers=0)
    iterator = iter(loader)
    started = time.time()
    for step in range(1,args.steps+1):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        batch = {k:v.to(args.device) for k,v in batch.items()}
        policy.train(); optimizer.zero_grad(set_to_none=True)
        loss, details = policy(batch)
        if not torch.isfinite(loss): raise RuntimeError("Nonfinite ACT loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(),optim_config.grad_clip_norm,error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            report = {"step":step,"loss":float(loss.detach()),"elapsed_s":time.time()-started,"status":"running"}
            print(json.dumps(report),flush=True)
            (args.out / "status.json").write_text(json.dumps(report))
        if step % args.save_every == 0 or step == args.steps:
            policy.eval(); error = count = 0
            with torch.no_grad():
                for b in val:
                    b = {k:v.to(args.device) for k,v in b.items()}
                    predicted = policy.predict_action_chunk(b)
                    mask = ~b["action_is_pad"]
                    error += float((predicted-b["action"]).abs()[mask].sum())
                    count += int(mask.sum())*39
            report = {"step":step,"validation_normalized_l1":error/count,"elapsed_s":time.time()-started,"autonomous_evaluation":False,"status":"completed" if step==args.steps else "running"}
            tmp = args.out / "checkpoint.partial"
            torch.save({"model":policy.state_dict(),"stats":stats,"step":step,"metadata":metadata},tmp)
            tmp.replace(args.out/f"step-{step:06d}.pt")
            with (args.out/"metrics.jsonl").open("a") as f: f.write(json.dumps(report)+"\n")
            (args.out/"status.json").write_text(json.dumps(report)); print(json.dumps(report),flush=True)


if __name__ == "__main__": main()

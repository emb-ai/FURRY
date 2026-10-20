"""Run one isolated smoke + training; refuses an occupied GPU at startup."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(); p.add_argument("kind",choices=["epsilon","sample","act","groot"]); p.add_argument("--gpu",required=True)
    args=p.parse_args()
    report=ROOT/f"job-{args.kind}.json"
    def status(state, **kw):
        value={"kind":args.kind,"status":state,"gpu":args.gpu,"pid":os.getpid(),"updated":time.time(),**kw}
        report.write_text(json.dumps(value,indent=2)); print(json.dumps(value),flush=True)
    raw=subprocess.check_output(["nvidia-smi","-i",args.gpu,"--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True)
    memory,util=[int(x.strip()) for x in raw.split(',')]
    if memory>500 or util>10:
        status("blocked_gpu_busy",memory_mib=memory,utilization=util); return
    verification=json.loads((ROOT/"transfer-verified.json").read_text())
    assert verification["verified"]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,OMP_NUM_THREADS="8",MKL_NUM_THREADS="8",WANDB_MODE="disabled",TOKENIZERS_PARALLELISM="false")
    data=str(ROOT/"cup44-no-shadow-causal.hdf5")
    if args.kind in ["epsilon","sample"]:
        base=[str(ROOT/"env-dp/bin/python"),str(ROOT/"train_dp.py"),"--data",data,"--weights",str(ROOT/"r3m-resnet18-encoder.pt"),"--prediction-type",args.kind]
        smoke=base+["--out",str(ROOT/f"smoke-dp-{args.kind}"),"--max-steps","3","--workers","0"]
        train=base+["--out",str(ROOT/f"dp-{args.kind}")]
        cwd=ROOT
    elif args.kind=="act":
        base=[str(ROOT/"env-act/bin/python"),str(ROOT/"train_act.py"),"--data",data]
        smoke=base+["--out",str(ROOT/"smoke-act"),"--steps","3","--save-every","3","--workers","0"]
        train=base+["--out",str(ROOT/"act-run")]
        cwd=ROOT
    else:
        cwd=ROOT/"Isaac-GR00T"
        base=[str(cwd/".venv/bin/python"),"gr00t/experiment/launch_finetune.py","--base-model-path","nvidia/GR00T-N1.7-3B","--dataset-path",str(ROOT/"groot-data/train"),"--embodiment-tag","NEW_EMBODIMENT","--modality-config-path",str(ROOT/"cup_groot_config.py"),"--num-gpus","1","--global-batch-size","32","--dataloader-num-workers","2"]
        smoke=base+["--output-dir",str(ROOT/"smoke-groot"),"--max-steps","3","--save-steps","3"]
        train=base+["--output-dir",str(ROOT/"groot-run"),"--max-steps","10000","--save-steps","1000"]
    for phase,command in [("smoke",smoke),("training",train)]:
        status(phase,command=command)
        result=subprocess.run(command,cwd=cwd,env=env)
        if result.returncode:
            status("failed",phase=phase,returncode=result.returncode); return
    status("completed",autonomous_evaluation=False)


if __name__=="__main__": main()

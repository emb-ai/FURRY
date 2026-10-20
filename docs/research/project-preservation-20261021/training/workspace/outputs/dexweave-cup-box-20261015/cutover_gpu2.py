"""Stop only the identified old DexWeave after verified successful resume."""
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
record=ROOT/'migration-episode-1792020055691156.json'
info=json.loads(record.read_text())
assert not info['old_process_stopped'], 'Cutover already recorded'
out=Path(info['new_output'])
status=json.loads((out/'status.json').read_text())
assert status['status']=='running' and status['pid']==info['new_pid']
assert status['iteration']>=503 and status['actor_changed']
assert all(math.isfinite(x) for x in status['losses'].values())
verified=json.loads((out/'resume-verification.json').read_text())
assert all(verified[k] for k in ['policy_exact','optimizer_exact','normalizers_exact'])
assert verified['iteration']==500
import torch
checkpoint=out/'checkpoint-00501.pt'
saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert saved['iteration']==501
assert all(torch.isfinite(t).all() for t in saved['policy'].values() if t.is_floating_point())
old_pid=info['old_pid'];new_pid=info['new_pid']
oldcmd=(Path('/proc')/str(old_pid)/'cmdline').read_bytes().decode().split('\0')
newcmd=(Path('/proc')/str(new_pid)/'cmdline').read_bytes().decode().split('\0')
assert old_pid==703179 and 'dexcup.train' in oldcmd and 'data/episode-1792020055691156.npz' in oldcmd
assert 'dexcup.train_resume' in newcmd and str(out) in newcmd
gpu_rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
gpu_by_pid={int(pid.strip()):u.strip() for u,pid in (line.split(',') for line in gpu_rows.splitlines())}
assert gpu_by_pid[old_pid]=='GPU-9d79e8fc-4895-171e-768c-f63ec6f58f47'
assert gpu_by_pid[new_pid]==info['new_gpu_uuid']
old_dir=ROOT/'train-episode-1792020055691156'
old_status=json.loads((old_dir/'status.json').read_text())
info.update(status='cutover_requested',source_last_status=old_status,destination_at_cutover=status,
            source_unsaved_iterations=old_status['iteration']-verified['iteration'],signal_pid=old_pid,signal='SIGTERM')
record.write_text(json.dumps(info,indent=2))
os.kill(old_pid,signal.SIGTERM)
for _ in range(30):
    stat=Path('/proc')/str(old_pid)/'stat'
    if not stat.exists() or stat.read_text().split(') ')[1][0]=='Z': break
    time.sleep(1)
else:
    raise RuntimeError('Old process did not exit after SIGTERM; no other process signalled')
info.update(status='migrated',old_process_stopped=True,completed_at=time.time(),foreign_processes_modified=False)
record.write_text(json.dumps(info,indent=2))
old_status.update(status='migrated',migration_record=str(record),continued_in=str(out),continued_pid=new_pid)
temp=old_dir/'migration-status.tmp';temp.write_text(json.dumps(old_status,indent=2));temp.replace(old_dir/'status.json')
print(json.dumps(info,indent=2))

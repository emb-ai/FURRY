import json,math,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent
reset=json.loads((p/'reset-verification.json').read_text())
assert reset['all_checks_passed'] and reset['physical_state_from_reference'] is False
for k in ['max_joint_velocity','max_root_velocity','history_abs_max','previous_action_abs_max','physics_time','control_step']:assert reset[k]==0,(k,reset[k])
spec=reset['reset_spec']
for root in reset['root_state']:
 assert max(abs(a-b) for a,b in zip(root,spec['root']+[0.]*6))<1e-6
for q in reset['joint_positions']:
 assert max(abs(a-spec['joints'][name]) for a,name in zip(q,reset['joint_names']))<1e-6
rollout_path=next(p.glob('rollout-*.json'));rollout=json.loads(rollout_path.read_text());rows=rollout['rows'];r=rollout['summary']
task=json.loads((p/'initial-states.json').read_text())['task'];g=task['collection_goal_0'];fallheight,fallup=task['collection_fall']
assert len(r['seeds'])==10 and len(set(r['seeds']))==10
assert rows and all(len(x['cup'])==10 for x in rows)
results=[]
for i,seed in enumerate(r['seeds']):
 firstfall=None;maxlift=0;inside_count=0;lifted=False;settled=0.;candidate=None
 for row in rows:
  vals=row['cup'][i]+row['cup_state'][i]+[row['root_z'][i],row['upright'][i]]
  assert all(math.isfinite(x) for x in vals)
  if row['root_z'][i]<fallheight or row['upright'][i]<fallup:
   firstfall=row['t'];break
  cup=row['cup'][i];lift=cup[2]-r['random_starts'][i][2];maxlift=max(maxlift,lift);lifted=lifted or lift>.045
  inside=all(abs(cup[k]-g[k])<g[k+3] for k in range(3));inside_count+=int(inside)
  vel=row['cup_state'][i][7:13];slow=math.sqrt(sum(x*x for x in vel[:3]))<.06 and math.sqrt(sum(x*x for x in vel[3:]))<.25
  settled=settled+.02 if lifted and inside and slow else 0
  if settled>=.5 and candidate is None:candidate=row['t']
 assert firstfall==r['first_task_fall_s'][i]
 assert abs(maxlift-r['classification'][i]['max_lift_before_fall_m'])<1e-8
 assert candidate==r['classification'][i]['candidate_goal_settle_s']
 results.append({'seed':seed,'first_fall_s':firstfall,'max_lift_before_end_m':maxlift,'goal_samples':inside_count,'candidate_goal_settle_s':candidate,'outcome':'requires_contact_verification' if candidate else ('fall' if firstfall else 'timeout')})
report={'checkpoint_iteration':r['iteration'],'checkpoint_sha256':r['sha256'],'protocol':'Native DexWeave neutral reset mapped to IsaacLab training physics; no demo physical state; zero velocities and control buffers; reference retained only as policy input','n':10,'successes':0 if all(x['candidate_goal_settle_s'] is None for x in results) else None,'falls':sum(x['first_fall_s'] is not None for x in results),'timeouts':sum(x['outcome']=='timeout' for x in results),'errors':[],'reset_checks_passed':True,'wall_seconds':r['elapsed_s'],'simulated_seconds':r['duration_s'],'results':results,'rollout_sha256':hashlib.sha256(rollout_path.read_bytes()).hexdigest(),'success_proof':'No trial met necessary task-success conditions' if all(x['candidate_goal_settle_s'] is None for x in results) else 'Contact-release verification required for candidates'}
(p/'verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))

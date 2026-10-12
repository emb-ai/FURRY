from evaluate import summarize
from pathlib import Path
import json,argparse,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args();out=args.root
labels={'twist2':'TWIST2 +500','homie':'HOMIE v2','sonic':'SONIC VR3 + planner'};colors={'twist2':'#737c8c','homie':'#087f8c','sonic':'#d36719'}
report={};loaded={}
for group,n in [('validation',12),('live',2)]:
 report[group]={};loaded[group]={}
 for backend in labels:
  paths=sorted((out/'final'/group).glob(backend+'-*.json'));paths=[p for p in paths if not p.name.endswith('.rows.json')]
  assert len(paths)==n,(group,backend,len(paths))
  runs=[json.loads(p.read_text()) for p in paths]
  rows={r['clip']:json.loads(p.with_suffix('.rows.json').read_text()) for p,r in zip(paths,runs)};loaded[group][backend]=rows
  joined=[r for values in rows.values() for r in values];summary=summarize(joined)
  # Path statistics must not connect the ends of different clips.
  for k in ['reference_path_m','actual_path_m']:summary[k]=sum(r['metrics'][k] for r in runs)
  summary['directed_progress_ratio']=sum(r['metrics']['directed_progress_ratio']*r['metrics']['reference_path_m'] for r in runs)/max(summary['reference_path_m'],1e-9)
  summary.update(clips=n,falls=sum(r['failure']=='fall' for r in runs),failures=[(r['clip'],r['failure']) for r in runs if r['failure']],seconds=sum(r['simulated_seconds'] for r in runs),inference_ms_median=float(np.median([r['inference_ms_median'] for r in runs])))
  report[group][backend]=summary
 # No fall truncation in these runs; explicitly verify matched lengths.
 for clip in loaded[group]['twist2']:
  lengths=[len(loaded[group][b][clip]) for b in labels];assert len(set(lengths))==1,(clip,lengths)
report['limitations']=['Diagnostic previously reused validation, not independent test.','Two live recordings are new counterfactual simulations, not headset screen captures.','No measurement of true human leg intent. Three-point targets come from common robot-scaled GMR FK.','HOMIE and SONIC retain respective author PD and motor dynamics; common scene, contact geometry, 50Hz control, 1000Hz physics.','Trip candidates are unlabelled 50Hz contact proxies. Ordinary braking touch-downs must not be interpreted as confirmed stumbles.','Single planner seed 11 in primary comparison; no statistical generalization claim.','SONIC wrapper replans synchronously at 5Hz with world-frame commands; this is a desktop adapter, not a timing-identical reproduction of asynchronous author deployment.','No Quest installation or Android performance acceptance.']
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# HOMIE / SONIC on FURRY clips','', 'All numbers below are counterfactual MuJoCo simulations, with catch-up enabled. Baseline: TWIST2 quest-111586-500 at 50 Hz.','']
for group in ['validation','live']:
 lines+=['## '+group,'','| Mode | Root XY mean / p95, cm | Slip, cm/s | Low moving foot, % | Wrist world / root-local, cm | Torso rotation, deg | Upward contact candidates | Falls |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for b,label in labels.items():
  r=report[group][b];lines.append(f"| {label} | {100*r['root_xy_m']['mean']:.1f} / {100*r['root_xy_m']['p95']:.1f} | {100*r['contact_slip_m_s']['mean']:.2f} | {100*r['low_moving_foot_fraction']['mean']:.1f} | {100*r['wrist_world_mean_m']['mean']:.1f} / {100*r['wrist_local_mean_m']['mean']:.1f} | {r['torso_orientation_deg']['mean']:.1f} | {r['upward_contact_events']} | {r['falls']}/{r['clips']} |")
 lines+=['',f"Duration per mode: {report[group]['twist2']['seconds']:.2f} s. Frame counts match clip-by-clip; no fall-conditioned truncation.",'']
lines+=['## Metric definitions','','- Root XY: error against the retargeted reference path, without post-hoc alignment. Live clips additionally report ego-camera XY error against recorded HMD goals in results.json.','- Slip: load-weighted tangential speed at actual sole-floor contact points, including angular foot motion; zero when unloaded. Compare together with contact and low-foot diagnostics.','- Low moving foot: fraction of foot-samples with exact sphere-sole clearance <1cm and horizontal foot-body speed >0.15m/s; normal landing/take-off may contribute.','- Braking contact: new foot load >10N, previous speed >0.15m/s, reduction >0.08m/s in 20ms, 150ms event debounce. This includes normal touchdowns.','- Upward contact candidate: braking contact plus previous upward foot-body speed >0.04m/s and sole gap >3mm. It is a narrow detector, not a complete or validated count of stumbles.','- Upper-body metrics: common GMR targets, world and full root-local positions; quaternion geodesic wrist/torso angles and 17 upper-body joint RMSE.','', '## Limitations','']+['- '+x for x in report['limitations']]
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
for row,group in enumerate(['validation','live']):
 for col,(key,title,scale) in enumerate([('root_xy_m','Trajectory error (cm)',100),('contact_slip_m_s','Contact sliding (cm/s)',100),('wrist_local_mean_m','Root-local wrist error (cm)',100)]):
  ax=axes[row,col];vals=[report[group][b][key]['mean']*scale for b in labels];bars=ax.bar(range(3),vals,color=[colors[b] for b in labels]);ax.set_xticks(range(3),['TWIST2 +500','HOMIE v2','SONIC']);ax.set_title(group+': '+title);ax.bar_label(bars,fmt='%.2f');ax.set_ylim(0,max(vals)*1.2);ax.grid(axis='y',alpha=.2)
fig.suptitle('Same clips and scene; autonomous legs, common upper-body targets\n12 diagnostic clips + 2 recorded live sequences; no headset deployment')
fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
for row,clip in enumerate(sorted(loaded['live']['twist2'])):
 r=loaded['live']['twist2'][clip];axes[row,0].plot([x['ref_x'] for x in r],[x['ref_y'] for x in r],'k--',label='Reference')
 for b in labels:
  rows=loaded['live'][b][clip];t=[r['time'] for r in rows]
  axes[row,0].plot([r['root_x'] for r in rows],[r['root_y'] for r in rows],color=colors[b],label=labels[b]);axes[row,1].plot(t,[r['root_xy_m'] for r in rows],color=colors[b]);axes[row,2].plot(t,[r['wrist_local_mean_m'] for r in rows],color=colors[b])
 for col,title in enumerate(['XY path (m)','Root tracking error (m)','Root-local wrist error (m)']):axes[row,col].set_title(clip+'\n'+title);axes[row,col].grid(alpha=.2)
 axes[row,0].axis('equal');axes[row,0].legend(fontsize=8)
fig.savefig(out/'live-trajectories.png',dpi=150);plt.close(fig)
print(json.dumps(report,indent=2))
# Optional second planner seed; deterministic controllers are not duplicated.
if (out/'seed22/live').exists():
 repeat={}
 for group,n in [('validation',12),('live',2)]:
  paths=sorted((out/'seed22'/group).glob('*.rows.json'))
  if len(paths)!=n:raise ValueError('Incomplete seed22 comparison')
  rows=[r for p in paths for r in json.loads(p.read_text())];summary=summarize(rows)
  runs=[json.loads(p.with_name(p.name.replace('.rows.json','.json')).read_text()) for p in paths]
  summary['falls']=sum(r['failure']=='fall' for r in runs)
  for k in ['reference_path_m','actual_path_m']:summary[k]=sum(r['metrics'][k] for r in runs)
  summary['directed_progress_ratio']=sum(r['metrics']['directed_progress_ratio']*r['metrics']['reference_path_m'] for r in runs)/max(summary['reference_path_m'],1e-9)
  repeat[group]=summary
 report['sonic_planner_seed22']=repeat
 report['limitations']=[s for s in report['limitations'] if not s.startswith('Single planner seed')]+['Primary SONIC table uses seed11; seed22 checks planner sampling sensitivity, not training variance.']
 (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');(out/'seed22-summary.json').write_text(json.dumps(repeat,indent=2)+'\n')
 p=out/'REPORT.md';s=p.read_text().replace('- Single planner seed 11 in primary comparison; no statistical generalization claim.','- Main table uses planner seed11; seed22 checks planner sampling sensitivity, not training variance.')
 s+='\n## SONIC planner seed sensitivity\n\n| Set | Seed | Root XY mean, cm | Slip, cm/s | Upward contact candidates | Falls |\n|---|---:|---:|---:|---:|---:|\n'
 for group in ['validation','live']:
  for seed,d in [(11,report[group]['sonic']),(22,repeat[group])]:s+=f"| {group} | {seed} | {100*d['root_xy_m']['mean']:.1f} | {100*d['contact_slip_m_s']['mean']:.2f} | {d['upward_contact_events']} | {d['falls']} |\n"
 s+='\nContact-candidate snapshots were inspected. They include edge contacts/rocking and do not establish a human-labelled stumble count; see contact-candidates.png. Event thresholds were clarified during this pilot, not pre-registered acceptance thresholds.\n'
 p.write_text(s)

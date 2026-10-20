from pathlib import Path
import sys,json,collections,hashlib
import numpy as np,mujoco
O=Path(__file__).resolve().parent;P=O.parent/'rubik-table-rest-20261015/project';sys.path.insert(0,str(P))
from g1_sim.visual_dataset import inspect_episode,read_json
R=O.parent/'task-demo-audit-20261015/recordings';rows=[];seen=set();families={}
for p in sorted(R.glob('episode-*')):
 s=read_json(p/'session_config.json');row={'episode':p.name,'scene':s['scene']}
 try:
  e=inspect_episode(p)
  if e['identity'] in seen:raise ValueError('Duplicate content')
  seen.add(e['identity']);family=json.dumps(e['family'],sort_keys=True);key=hashlib.sha256(family.encode()).hexdigest()[:12]
  familyrow=families.setdefault(key,{'family':e['family'],'episodes':[],'seconds':0.,'samples':0,'layouts':[]})
  seconds=float(e['grid'][-1]-e['grid'][0]);familyrow['episodes'].append(p.name);familyrow['seconds']+=seconds;familyrow['samples']+=len(e['grid'])
  m=mujoco.MjModel.from_binary_path(str(p/'learning-model.mjb'));q=int(m.joint('cup_free').qposadr[0]);i=e['start'];f=e['frames'];start=[float(f[f'qpos_{q+k}'][i]) for k in range(3)]
  b=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'cup_goal_marker')
  if b>=0:
   mocap=m.body_mocapid[b];goal=[float(e['scene'][f'mocap_pos_{3*mocap+k}'][i]) for k in range(3)]
  else:goal=m.numeric('collection_goal_0').data[:3].tolist()
  layout={'start':start,'goal':goal,'distance_m':float(np.linalg.norm(np.array(start[:2])-goal[:2]))};familyrow['layouts'].append(layout)
  row.update(accepted=True,family_id=key,seconds=seconds,samples=len(e['grid']),identity=e['identity'],layout=layout)
 except Exception as ex:row.update(accepted=False,rejection=str(ex))
 rows.append(row)
for key,g in families.items():
 g['count']=len(g['episodes']);g['distance_range_m']=[min(x['distance_m'] for x in g['layouts']),max(x['distance_m'] for x in g['layouts'])]
 g['long_at_least_32cm']=sum(x['distance_m']>=.32 for x in g['layouts'])
 members=[r for r in rows if r.get('family_id')==key];nvalid=max(1,round(len(members)*.1));valid={r['episode'] for r in sorted(members,key=lambda r:r['identity'])[:nvalid]}
 g['train']=[r['episode'] for r in members if r['episode'] not in valid];g['valid']=sorted(valid)
 (O/f'family-{key}.json').write_text(json.dumps(g,ensure_ascii=False,indent=2)+'\n')
summary={'total':len(rows),'scenes':dict(collections.Counter(r['scene'] for r in rows)),'accepted':sum(r['accepted'] for r in rows),'accepted_seconds':sum(r.get('seconds',0) for r in rows),'rejections':dict(collections.Counter(r['rejection'] for r in rows if not r['accepted'])),'families':families,'episodes':rows}
(O/'audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('episodes','families')},indent=2));print(json.dumps([{k:v for k,v in g.items() if k not in ('layouts','episodes','family','train','valid')}|{'id':i,'train':len(g['train']),'valid':len(g['valid'])} for i,g in families.items()],indent=2))

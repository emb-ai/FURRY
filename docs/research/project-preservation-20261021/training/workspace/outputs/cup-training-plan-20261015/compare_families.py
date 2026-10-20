from pathlib import Path
import json,numpy as np,mujoco
O=Path(__file__).resolve().parent;R=O.parent/'task-demo-audit-20261015/recordings';a=json.loads((O/'audit.json').read_text());gs=list(a['families'].values());models=[mujoco.MjModel.from_binary_path(str(R/g['episodes'][0]/'learning-model.mjb')) for g in gs];x,y=models;diff=[]
for obj,fields in [('body',['pos','quat','mass','inertia','ipos','iquat']),('joint',['pos','axis','range','limited','stiffness']),('actuator',['ctrlrange','forcerange','gear','gainprm','biasprm','dynprm']),('geom',['type','size','pos','quat','friction','solref','solimp','contype','conaffinity','condim'])]:
 num={'body':'nbody','joint':'njnt','actuator':'nu','geom':'ngeom'}[obj];nx={getattr(x,obj)(i).name:i for i in range(getattr(x,num))};ny={getattr(y,obj)(i).name:i for i in range(getattr(y,num))}
 for name in sorted(set(nx)|set(ny)):
  if not name:continue
  if name not in nx or name not in ny:diff.append({'type':obj,'name':name,'change':'added' if name not in nx else 'removed'});continue
  if obj=='geom' and not (x.geom_contype[nx[name]] or x.geom_conaffinity[nx[name]] or y.geom_contype[ny[name]] or y.geom_conaffinity[ny[name]]):continue
  for field in fields:
   xx=getattr(getattr(x,obj)(name),field);yy=getattr(getattr(y,obj)(name),field)
   if not np.array_equal(xx,yy):diff.append({'type':obj,'name':name,'field':field,'old':np.asarray(xx).tolist(),'new':np.asarray(yy).tolist()})
for field in ['dof_armature','dof_damping','dof_frictionloss']:
 if not np.array_equal(getattr(x,field),getattr(y,field)):diff.append({'type':'dof','field':field})
options={f:[np.asarray(getattr(m.opt,f)).tolist() for m in models] for f in ['timestep','solver','iterations','noslip_iterations','enableflags','disableflags','gravity','integrator']}
r={'dimensions':[{k:getattr(m,k) for k in ['nq','nv','nu','nbody','ngeom','nmocap','nuserdata']} for m in models],'differences':diff,'options':options,'family_ids':list(a['families'])}
(O/'family-compatibility.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))

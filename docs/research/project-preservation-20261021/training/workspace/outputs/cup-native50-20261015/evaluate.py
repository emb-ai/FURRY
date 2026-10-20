from pathlib import Path
import ctypes,json,time,argparse,os
import numpy as np,torch,mujoco as mj
from PIL import Image
O=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--hz',type=int,choices=[30,50],default=50);p.add_argument('--device',default='cuda');p.add_argument('--limit',type=float,default=90);p.add_argument('--cases',type=int,default=10);args=p.parse_args();
if (args.out/'summary.json').exists():
 done=json.loads((args.out/'summary.json').read_text());assert done['n']==args.cases and not done['errors'] and done['action_hz']==args.hz and done['sim_limit_s']==args.limit and Path(done['weights']).resolve()==args.weights.resolve();print('Reusing completed identical evaluation',str(args.out));raise SystemExit()
args.out.mkdir(exist_ok=False,parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
import onnxruntime
ctypes.CDLL(str(next((Path(onnxruntime.__file__).parent/'capi').glob('libonnxruntime.so.*'))),mode=ctypes.RTLD_GLOBAL)
print('loading native bridge',flush=True)
lib=ctypes.CDLL(str(O/'bridge.so'));ptr=ctypes.c_void_p;dp=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS');fp=np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS')
lib.create.argtypes=[ctypes.c_char_p];lib.create.restype=ptr;lib.destroy.argtypes=[ptr];lib.last_error.restype=ctypes.c_char_p;lib.read_state.argtypes=[ptr,dp,dp,dp,dp,dp];lib.step.argtypes=[ptr,fp,ctypes.c_int];lib.reset_random.argtypes=[ptr,ctypes.c_uint];lib.fresh_reset_flags.argtypes=[ptr]
print('loading ACT',flush=True)
model=torch.jit.load(str(args.weights/'act-inference.pt'),map_location=args.device).eval();stats=json.loads((args.weights/'stats.json').read_text());stats={k:{kk:np.array(v,np.float32) for kk,v in s.items()} for k,s in stats.items()};m=mj.MjModel.from_binary_path(str(O/'episode/learning-model.mjb'));d=mj.MjData(m);r=mj.Renderer(m,height=360,width=640);cam=m.camera('ego').id;cq=int(m.joint('cup_free').qposadr[0]);m.light_castshadow[:]=0;rows=[];cases=json.loads((O/'scenes.json').read_text())['cases'][:args.cases]
print('renderer and policy ready',flush=True)
for case in cases:
 seed=case['seed'];torch.manual_seed(seed);np.random.seed(seed);print('creating native reset',seed,flush=True);w=lib.create(str(O/'episode').encode());assert w,lib.last_error().decode();assert lib.reset_random(w,seed)==0 and lib.fresh_reset_flags(w)==63
 q=np.zeros(m.nq);v=np.zeros(m.nv);u=np.zeros(m.nuserdata);mp=np.zeros(m.nmocap*3);mq=np.zeros(m.nmocap*4)
 def read():
  lib.read_state(w,q,v,u,mp,mq);d.qpos[:]=q;d.qvel[:]=v;d.userdata[:]=u;d.mocap_pos.flat[:]=mp;d.mocap_quat.flat[:]=mq;mj.mj_forward(m,d)
 read()
 for actual,key in [(q,'qpos'),(v,'qvel'),(u,'userdata'),(mp,'mocap_pos'),(mq,'mocap_quat')]:np.testing.assert_allclose(actual,case[key],atol=1e-12,rtol=0)
 start=time.time();startcup=q[cq:cq+3].copy();goal=mp[:3].copy();previous=np.zeros(39,np.float32);gaze=np.zeros(2,np.float32);maxlift=0;maxtilt=0;commands=[];qs=[];vs=[];trace=[];elapsed=0;outcome=0;initial_rgb=None
 try:
  for k in range(int(args.limit*args.hz)):
   if k%8==0:
    yaw,pitch=gaze;cy,sy=np.cos(yaw),np.sin(yaw);cp,sp=np.cos(pitch),np.sin(pitch);R=d.cam_xmat[cam].reshape(3,3)@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])@np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]]);r.update_scene(d,camera=cam)
    for c in r.scene.camera:c.pos[:]=d.cam_xpos[cam];c.forward[:]=-R[:,2];c.up[:]=R[:,1]
    r.scene.flags[mj.mjtRndFlag.mjRND_SHADOW]=0;rgb=np.asarray(Image.fromarray(r.render()).resize((224,224),Image.Resampling.BILINEAR))
    if initial_rgb is None:initial_rgb=rgb.copy()
    im=rgb.transpose(2,0,1).astype(np.float32)/255;im=(im-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None];s=stats['obs/previous_command'];x=(previous-s['mean'])/s['std']
    if 'obs/proprio' in stats:
     from proprio import features
     x=np.concatenate([x,(features(q,v)-stats['obs/proprio']['mean'])/stats['obs/proprio']['std']])
    with torch.inference_mode():actions=model(torch.from_numpy(im)[None].to(args.device),torch.from_numpy(x)[None].to(args.device)).cpu().numpy()[0]*stats['actions']['std']+stats['actions']['mean']
   a=np.ascontiguousarray(actions[k%8],np.float32);assert np.isfinite(a).all();n=round((k+1)*1000/args.hz)-round(k*1000/args.hz);outcome=lib.step(w,a,n);read();previous=a.copy();gaze=a[37:39].copy();elapsed+=n/1000;cup=q[cq:cq+3];qt=q[cq+3:cq+7];tilt=float(np.rad2deg(np.arccos(np.clip(1-2*(qt[1]**2+qt[2]**2),-1,1))));maxlift=max(maxlift,float(cup[2]-startcup[2]));maxtilt=max(maxtilt,tilt);qs.append(q.copy());vs.append(v.copy());commands.append(a);trace.append([elapsed,*cup,tilt,*u])
   if outcome:break
  row={'seed':seed,'success':outcome==1,'outcome':outcome,'sim_s':elapsed,'wall_s':time.time()-start,'lift_m':maxlift,'max_cup_tilt_deg':maxtilt,'final_cup_tilt_deg':tilt,'goal_xy_error_m':float(np.linalg.norm(cup[:2]-goal[:2])),'error':lib.last_error().decode() if outcome<0 else None};rows.append(row);np.savez_compressed(args.out/f'{seed}.npz',qpos=qs,qvel=vs,actions=commands,trace=trace,mocap_pos=mp,mocap_quat=mq,initial_rgb=initial_rgb);(args.out/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
 finally:lib.destroy(w)
r.close();summary={'status':'completed','successes':sum(x['success'] for x in rows),'n':len(rows),'errors':[x['error'] for x in rows if x['error']],'sim_limit_s':args.limit,'action_hz':args.hz,'execution_steps':8,'protocol':'10 identical full native resets; neutral gaze; no demonstration prefix','platform':'Linux native MuJoCo3.3.7, archived HOMIE/model, Mesa software EGL rendering, CUDA inference GPU6','weights':str(args.weights),'results':rows};(args.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='results'}),flush=True)

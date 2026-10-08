"""Retarget RGB-D landmarks to an operator-sized human MJCF using upstream GMR tasks.

Offline visualization only. Local participant input/output must stay outside Git.
"""
from pathlib import Path
import argparse,importlib.util,sys,types,json,xml.etree.ElementTree as E,time
import numpy as np
import mujoco,mink
from scipy.spatial.transform import Rotation

NAMES=['pelvis','chest','head','lhip','rhip','lknee','rknee','lankle','rankle','lshoulder','rshoulder','lelbow','relbow','lwrist','rwrist']
COCO={'lhip':11,'rhip':12,'lknee':13,'rknee':14,'lankle':15,'rankle':16,'lshoulder':5,'rshoulder':6,'lelbow':7,'relbow':8,'lwrist':9,'rwrist':10}
LINKS=[('pelvis','chest'),('chest','head'),('pelvis','lhip'),('pelvis','rhip'),('lhip','lknee'),('rhip','rknee'),('lknee','lankle'),('rknee','rankle'),('chest','lshoulder'),('chest','rshoulder'),('lshoulder','lelbow'),('rshoulder','relbow'),('lelbow','lwrist'),('relbow','rwrist')]

def floor_frame(directory):
    """Estimate gravity from the observed floor, never from the operator's posture."""
    files=sorted(Path(directory).glob('frame-*.npz'))
    clouds=[]
    for path in [files[i] for i in np.linspace(0,len(files)-1,8,dtype=int)]:
        with np.load(path) as a:
            depth=a['depth_m'];fx,fy,cx,cy=a['intrinsics']
            if np.any(a['distortion_coeffs']):raise ValueError('Nonzero distortion needs SDK deprojection')
            yy,xx=np.mgrid[int(depth.shape[0]*.63):depth.shape[0]:8,0:depth.shape[1]:8]
            z=depth[yy,xx];p=np.stack([(xx-cx)*z/fx,(yy-cy)*z/fy,z],-1).reshape(-1,3)
            clouds.append(p[np.isfinite(p).all(1)&(p[:,2]>.3)&(p[:,2]<5)])
    p=np.concatenate(clouds);rng=np.random.default_rng(8);best=[]
    for _ in range(700):
        s=p[rng.choice(len(p),3,False)];n=np.cross(s[1]-s[0],s[2]-s[0]);norm=np.linalg.norm(n)
        if norm<1e-8:continue
        n/=norm
        if abs(n[1])<.85:continue
        offset=np.sum(s[0]*n);idx=np.flatnonzero(abs(np.sum(p*n,axis=1)-offset)<.018)
        if len(idx)>len(best):best=idx
    if len(best)<.2*len(p):raise ValueError('Floor plane not sufficiently supported')
    q=p[best];centre=q.mean(0);_,_,v=np.linalg.svd(q-centre,full_matrices=False);up=v[-1]
    if up[1]>0:up=-up
    right=np.array([1.,0,0]);right-=up*np.dot(right,up);right/=np.linalg.norm(right)
    forward=np.cross(up,right);basis=np.stack([right,forward,up]);offset=-np.dot(up,centre)
    return basis,offset,{'normal_camera':up.tolist(),'offset_m':float(offset),'inlier_fraction':len(best)/len(p),'residual_p95_m':float(np.quantile(abs(np.sum(q*up,axis=1)+offset),.95)),'camera_tilt_deg':float(np.degrees(np.arccos(-up[1])))}

def landmarks(row):
    pts={}
    for k,j in row.get('depth_points',{}).items():
        if j['conf']>=.55 and np.isfinite(j['p']).all():pts[int(k)]=np.array(j['p'],float)
    p={n:pts[i] for n,i in COCO.items() if i in pts}
    for n,a,b in [('pelvis','lhip','rhip'),('chest','lshoulder','rshoulder')]:
        if a in p and b in p:p[n]=(p[a]+p[b])/2
    if 3 in pts and 4 in pts:p['head']=(pts[3]+pts[4])/2
    elif 0 in pts:p['head']=pts[0]+[0,0,.075] # approximate face-to-head centre
    return p

def reject_occluded_middles(p, lengths):
    """Reject a middle-joint depth when both bones fail but endpoints are feasible.

    This does not supply a replacement measurement or anchor a foot to the floor.
    GMR can only infer the unobserved joint from its remaining tasks and history.
    """
    p=p.copy();rejected=[]
    for side in ['l','r']:
        for a,b,c,l1,l2 in [('hip','knee','ankle','thigh','shin'),('shoulder','elbow','wrist','upper_arm','forearm')]:
            a,b,c=[side+n for n in [a,b,c]]
            if not all(n in p for n in [a,b,c]):continue
            u,v=lengths[l1],lengths[l2]
            da=np.linalg.norm(p[a]-p[b]);db=np.linalg.norm(p[b]-p[c]);dc=np.linalg.norm(p[a]-p[c])
            if abs(u-v)-.06<=dc<=u+v+.06 and abs(da-u)>max(.10,.25*u) and abs(db-v)>max(.10,.25*v):
                del p[b];rejected.append(b)
    return p,rejected

def estimate(rows,height):
    values=[landmarks(r) for r in rows]
    def length(edges):
        x=[np.linalg.norm(p[a]-p[b]) for p in values for a,b in edges if a in p and b in p]
        if len(x)<15:raise ValueError('Insufficient calibration: '+str(edges))
        return float(np.median(x))
    d={'hip_width':length([('lhip','rhip')]),'shoulder_width':length([('lshoulder','rshoulder')]),
       'thigh':length([('lhip','lknee'),('rhip','rknee')]),'shin':length([('lknee','lankle'),('rknee','rankle')]),
       'upper_arm':length([('lshoulder','lelbow'),('rshoulder','relbow')]),'forearm':length([('lelbow','lwrist'),('relbow','rwrist')]),
       'torso':length([('pelvis','chest')]),'neck_head':length([('chest','head')])}
    # Bilateral symmetric approximation with the supplied standing height.
    scale=(height-.16)/(d['thigh']+d['shin']+d['torso']+d['neck_head'])
    return {k:v*scale for k,v in d.items()}

def make_model(d,path):
    root=E.Element('mujoco',model='operator_human');E.SubElement(root,'compiler',angle='radian');E.SubElement(root,'option',timestep='.01',gravity='0 0 0')
    default=E.SubElement(root,'default');E.SubElement(default,'geom',type='sphere',size='.015',mass='.1',contype='0',conaffinity='0')
    world=E.SubElement(root,'worldbody')
    def body(parent,name,pos=(0,0,0),joint=None):
        b=E.SubElement(parent,'body',name=name,pos=' '.join(map(str,pos)))
        if joint=='free':E.SubElement(b,'freejoint',name=name+'_joint')
        elif joint=='ball':E.SubElement(b,'joint',name=name+'_joint',type='ball',limited='false')
        elif joint: E.SubElement(b,'joint',name=name+'_joint',type='hinge',axis=joint,range='0 2.79',limited='true')
        E.SubElement(b,'geom');return b
    pelvis=body(world,'pelvis',joint='free')
    torso=body(pelvis,'torso_joint',joint='ball');chest=body(torso,'chest',(0,0,d['torso']))
    neck=body(chest,'neck_joint',joint='ball');body(neck,'head',(0,0,d['neck_head']))
    for side,sign in [('l',1),('r',-1)]:
        hip=body(pelvis,side+'hip',(sign*d['hip_width']/2,0,0),'ball')
        knee=body(hip,side+'knee',(0,0,-d['thigh']),'-1 0 0');body(knee,side+'ankle',(0,0,-d['shin']))
        shoulder=body(chest,side+'shoulder',(sign*d['shoulder_width']/2,0,0),'ball')
        elbow=body(shoulder,side+'elbow',(0,0,-d['upper_arm']),'1 0 0');body(elbow,side+'wrist',(0,0,-d['forearm']))
    E.ElementTree(root).write(path,encoding='unicode')

def main():
    a=argparse.ArgumentParser();a.add_argument('input');a.add_argument('output');a.add_argument('--height',type=float,default=1.75);a.add_argument('--calibration',nargs=2,type=float,default=[12,20]);a.add_argument('--profile');a.add_argument('--rgbd',required=True);args=a.parse_args()
    root=Path(__file__).resolve().parents[1];out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(l) for l in Path(args.input).read_text().splitlines()];start=rows[0]['t'];cal=[r for r in rows if args.calibration[0]*1000<=r['t']-start<args.calibration[1]*1000]
    d=json.loads(Path(args.profile).read_text()) if args.profile else estimate(cal,args.height)
    (out/'proportions.json').write_text(json.dumps(d,indent=2));make_model(d,out/'human.xml')
    pack=types.ModuleType('human_gmr_upstream');pack.__path__=[str(root/'vendor/GMR/general_motion_retargeting')];sys.modules[pack.__name__]=pack
    spec=importlib.util.spec_from_file_location(pack.__name__+'.motion_retarget',Path(pack.__path__[0])/'motion_retarget.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    cfg={'human_height_assumption':args.height,'ground_height':0,'human_root_name':'pelvis','robot_root_name':'pelvis','use_ik_match_table1':True,'use_ik_match_table2':True,'human_scale_table':{n:1 for n in NAMES}}
    for stage in [1,2]:cfg[f'ik_match_table{stage}']={n:[n,6 if n in ['pelvis','chest'] else (2 if stage==1 else 4),.6 if n=='pelvis' else 0,[0,0,0],[1,0,0,0]] for n in NAMES}
    (out/'gmr-human.json').write_text(json.dumps(cfg,indent=2));mod.ROBOT_XML_DICT['operator']=out/'human.xml';mod.IK_CONFIG_DICT['camera_landmarks']={'operator':out/'gmr-human.json'}
    g=mod.GeneralMotionRetargeting('camera_landmarks','operator',verbose=False);m=g.model;c=g.configuration
    cps=[landmarks(r) for r in cal];origin=np.median([p['pelvis'] for p in cps if 'pelvis'in p],axis=0)
    basis,floor_offset,floor_report=floor_frame(args.rgbd)
    (out/'floor.json').write_text(json.dumps(floor_report,indent=2))
    projected_origin=basis@origin
    def world(p):
        v=basis@p;return v-np.array([projected_origin[0],projected_origin[1],-floor_offset])
    first=next(r for r in rows if r['t']-start>=args.calibration[0]*1000 and 'pelvis' in landmarks(r))
    q=m.qpos0.copy();q[:3]=world(landmarks(first)['pelvis'])
    for j in range(m.njnt):
        if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE:q[m.jnt_qposadr[j]]=.15
    c.update(q)
    bodyids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in NAMES]
    bone_model=[{'name':m.body(i).name,'parent':int(m.body_parentid[i])-1,'p':m.body_pos[i].tolist()} for i in range(1,m.nbody)]
    frames=[];errors=[];times=[]
    for row in rows:
        if row['t']<first['t']:continue
        raw_p={n:world(v) for n,v in landmarks(row).items()}
        p,rejected=reject_occluded_middles(raw_p,d)
        if 'pelvis' not in p:continue
        rot=np.array([1.,0,0,0])
        if all(n in p for n in ['lhip','rhip','chest']):
            x=p['lhip']-p['rhip'];x/=max(np.linalg.norm(x),1e-8);z=p['chest']-p['pelvis'];z-=x*np.dot(z,x);z/=max(np.linalg.norm(z),1e-8);y=np.cross(z,x)
            rot=Rotation.from_matrix(np.stack([x,y,z],axis=1)).as_quat(scalar_first=True)
        # Set missing observations to current FK with zero task weight, never call them measured.
        human={n:[p.get(n,c.data.xpos[b].copy()),rot] for n,b in zip(NAMES,bodyids)}
        g.update_targets(human)
        for stage in [1,2]:
            for n,task in getattr(g,f'human_body_to_task{stage}').items():
                task.set_position_cost(cfg[f'ik_match_table{stage}'][n][1] if n in p else 0)
                task.set_orientation_cost(.6 if n=='pelvis' else 0)
        begin=time.perf_counter()
        # Upstream GMR tasks, explicit limits for the installed Mink API.
        # Kinematic replay only: no MuJoCo dynamics or foot contact controller.
        for tasks in [g.tasks1,g.tasks2]:
            for iteration in range(12):
                vel=mink.solve_ik(c,tasks,.01,'daqp',damping=.1,limits=g.ik_limits)
                c.integrate_inplace(vel,.01)
                if np.linalg.norm(vel)*.01<2e-4:break
        times.append((time.perf_counter()-begin)*1000)
        assert np.isfinite(c.q).all()
        residual=[np.linalg.norm(c.data.xpos[b]-p[n]) for n,b in zip(NAMES,bodyids) if n in p]
        errors.append(float(np.sqrt(np.mean(np.square(residual)))))
        locals=[]
        for b in range(1,m.nbody):
            parent=m.body_parentid[b];r=Rotation.from_quat(c.data.xquat[parent],scalar_first=True).inv()*Rotation.from_quat(c.data.xquat[b],scalar_first=True)
            locals.extend(r.as_quat(scalar_first=True).tolist())
        frames.append({'t':round((row['t']-first['t'])/1000,4),'root':np.round(c.data.xpos[1],5).tolist(),'rot':np.round(locals,6).tolist(),'raw':[np.round(raw_p[n],4).tolist() if n in raw_p else None for n in NAMES],'inferred':[NAMES.index(n) for n in NAMES if n not in p],'rejected':rejected,'err':round(errors[-1]*100,2)})
        if len(frames)%80==0:print('retargeted',len(frames),flush=True)
    payload={'kinematic_only':True,'floor':floor_report,'names':NAMES,'links':[[NAMES.index(a),NAMES.index(b)] for a,b in LINKS],'model':bone_model,'nodeBodies':[int(b)-1 for b in bodyids],'frames':frames,'profile':d,'height':args.height,'sourceStart':round((first['t']-start)/1000,3)}
    (out/'animation.json').write_text(json.dumps(payload,separators=(',',':')))
    metrics={'frames':len(frames),'duration':frames[-1]['t'],'position_rms_cm_p50_p95':np.quantile(np.array(errors)*100,[.5,.95]).tolist(),'solve_ms_p50_p95':np.quantile(times,[.5,.95]).tolist(),'upstream_gmr':True,'physical_simulation':False}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()

"""Export successful robot demonstrations and exact collision geometry, without altering recordings."""
import sys,json,hashlib,shutil
from pathlib import Path
import xml.etree.ElementTree as X
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation, Slerp

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent/'rubik-table-rest-20261015/project'
sys.path.insert(0,str(PROJECT))
from g1_sim.visual_dataset import inspect_episode,columns
from dexcup.body_spec import JOINTS as BODY,LINKS as BODY_LINKS
HANDS=tuple(f'{s}_hand_{p}_joint' for s in ('left','right') for p in ('thumb_0','thumb_1','thumb_2','middle_0','middle_1','index_0','index_1'))
JOINTS=BODY+HANDS
LINKS=BODY_LINKS+tuple(f'{s}_hand_{p}_finger_tip' for s in ('left','right') for p in ('thumb','middle','index'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def vec(v):return ' '.join(f'{x:.12g}' for x in v)
def rpy(q):return Rotation.from_quat(np.roll(q,-1)).as_euler('xyz')
def origin(node,pos,quat):X.SubElement(node,'origin',xyz=vec(pos),rpy=vec(rpy(quat)))
def name(m,kind,i):return mj.mj_id2name(m,kind,i)
def geometry(m,g,link,out):
    typ=int(m.geom_type[g]);size=m.geom_size[g];p=m.geom_pos[g];q=m.geom_quat[g]
    shapes=[(p,q,typ,size)]
    if typ==3: # URDF has no capsule: exactly equivalent cylinder + end spheres.
        R=Rotation.from_quat(np.roll(q,-1));shapes=[(p,q,5,size)]
        shapes += [(p+R.apply([0,0,s*size[1]]),q,2,size) for s in (-1,1)]
    for k,(p,q,t,s) in enumerate(shapes):
        c=X.SubElement(link,'collision',name=f'geom_{g}_{k}');origin(c,p,q);geom=X.SubElement(c,'geometry')
        if t==2:X.SubElement(geom,'sphere',radius=str(s[0]))
        elif t==5:X.SubElement(geom,'cylinder',radius=str(s[0]),length=str(2*s[1]))
        elif t==6:X.SubElement(geom,'box',size=vec(2*s))
        elif t==7:
            mid=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]];faces=m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid]+m.mesh_facenum[mid]]
            path=out/f'mesh_{mid}.stl'
            if not path.exists():
                with path.open('w') as f:
                    f.write('solid collision\n')
                    for tri in v[faces]:
                        f.write('facet normal 0 0 0\nouter loop\n')
                        for vert in tri:f.write('vertex '+vec(vert)+'\n')
                        f.write('endloop\nendfacet\n')
                    f.write('endsolid collision\n')
            X.SubElement(geom,'mesh',filename=path.name,scale='1 1 1')
        else:raise ValueError(f'Unsupported collision {t}')
def urdf(m,bodies,out,filename):
    root=X.Element('robot',name=filename.removesuffix('.urdf'))
    for b in bodies:
        bn=name(m,mj.mjtObj.mjOBJ_BODY,b);link=X.SubElement(root,'link',name=bn)
        if m.body_mass[b]>0:
            inert=X.SubElement(link,'inertial');origin(inert,m.body_ipos[b],m.body_iquat[b]);X.SubElement(inert,'mass',value=str(m.body_mass[b]))
            d=m.body_inertia[b];X.SubElement(inert,'inertia',ixx=str(d[0]),iyy=str(d[1]),izz=str(d[2]),ixy='0',ixz='0',iyz='0')
        else:
            # Preserve named FK markers without the URDF importer's unit-mass fallback.
            # Eight fixed marker links add only eight milligrams in total.
            inert=X.SubElement(link,'inertial');X.SubElement(inert,'mass',value='0.000001')
            X.SubElement(inert,'inertia',ixx='1e-10',iyy='1e-10',izz='1e-10',ixy='0',ixz='0',iyz='0')
        for g in range(m.body_geomadr[b],m.body_geomadr[b]+m.body_geomnum[b]):
            if m.geom_contype[g] or m.geom_conaffinity[g]:geometry(m,g,link,out)
        parent=int(m.body_parentid[b])
        if parent not in bodies:continue
        n=int(m.body_jntnum[b]);assert n<=1
        jid=int(m.body_jntadr[b])
        j=X.SubElement(root,'joint',name=name(m,mj.mjtObj.mjOBJ_JOINT,jid) if n else bn+'_fixed',type='revolute' if n else 'fixed')
        X.SubElement(j,'parent',link=name(m,mj.mjtObj.mjOBJ_BODY,parent));X.SubElement(j,'child',link=bn)
        origin(j,m.body_pos[b],m.body_quat[b])
        if n:
            assert int(m.jnt_type[jid])==3 and np.max(np.abs(m.jnt_pos[jid]))==0
            X.SubElement(j,'axis',xyz=vec(m.jnt_axis[jid]));limits=m.jnt_range[jid]
            X.SubElement(j,'limit',lower=str(limits[0]),upper=str(limits[1]),effort='200',velocity='50')
    X.indent(root);X.ElementTree(root).write(out/filename,encoding='unicode')
def main():
    out=ROOT/'data';out.mkdir(exist_ok=True);asset=out/'assets';asset.mkdir(exist_ok=True)
    raw=ROOT.parent/'task-demo-audit-20261015/recordings'
    accepted=[];rejected=[];modelhash=None
    for p in sorted(raw.glob('episode-*')):
        if json.loads((p/'session_config.json').read_text()).get('scene')!='wb_dex':continue
        try:e=inspect_episode(p)
        except (ValueError,KeyError) as ex:rejected.append({'episode':p.name,'reason':str(ex)});continue
        m=mj.MjModel.from_binary_path(str(p/'learning-model.mjb'))
        if modelhash is None:
            modelhash=sha(p/'learning-model.mjb');shutil.copy2(p/'learning-model.mjb',asset/'reference.mjb')
            urdf(m,list(range(1,54)),asset,'robot.urdf');urdf(m,[55],asset,'cup.urdf')
            statics=[]
            for b in (54,56):
                for g in range(m.body_geomadr[b],m.body_geomadr[b]+m.body_geomnum[b]):
                    assert int(m.geom_type[g])==6
                    R=Rotation.from_quat(np.roll(m.body_quat[b],-1));q=(R*Rotation.from_quat(np.roll(m.geom_quat[g],-1))).as_quat()
                    statics.append(dict(name=name(m,mj.mjtObj.mjOBJ_GEOM,g).replace('-','m').replace('.','_'),size=(2*m.geom_size[g]).tolist(),pos=(m.body_pos[b]+R.apply(m.geom_pos[g])).tolist(),quat=np.roll(q,1).tolist()))
            (asset/'statics.json').write_text(json.dumps(statics,indent=2))
        assert modelhash==sha(p/'learning-model.mjb')
        f=e['frames'];sl=slice(e['start'],e['end']+1);t=f['sim_time'][sl];grid=t[0]+np.arange(int(np.floor((t[-1]-t[0])*50))+1)/50
        q0=columns(f,'qpos',m.nq)[sl];v0=columns(f,'qvel',m.nv)[sl]
        q=np.stack([np.interp(grid,t,a) for a in q0.T],axis=1);v=np.stack([np.interp(grid,t,a) for a in v0.T],axis=1)
        for start in (3,53):q[:,start:start+4]=np.roll(Slerp(t,Rotation.from_quat(np.roll(q0[:,start:start+4],-1,axis=1)))(grid).as_quat(),1,axis=1)
        ji=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,n) for n in JOINTS];bi=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,n) for n in LINKS]
        poses=[];quats=[];lin=[];ang=[];obj=[];data=mj.MjData(m)
        for qi,vi in zip(q,v):
            data.qpos[:]=qi;data.qvel[:]=vi;mj.mj_forward(m,data)
            poses.append(data.xpos[bi].copy());quats.append(data.xquat[bi].copy());lv=[];av=[]
            for b in bi:
                vel=np.zeros(6);mj.mj_objectVelocity(m,data,mj.mjtObj.mjOBJ_BODY,b,vel,0);lv.append(vel[3:]);av.append(vel[:3])
            lin.append(lv);ang.append(av)
            vel=np.zeros(6);mj.mj_objectVelocity(m,data,mj.mjtObj.mjOBJ_BODY,55,vel,0);obj.append(np.r_[data.xpos[55],data.xquat[55],vel[3:],vel[:3]])
        path=out/(p.name+'.npz')
        np.savez_compressed(path,fps=50,joint_names=JOINTS,body_names=LINKS,joint_pos=q[:,m.jnt_qposadr[ji]],joint_vel=v[:,m.jnt_dofadr[ji]],body_pos_w=poses,body_quat_w=quats,body_lin_vel_w=lin,body_ang_vel_w=ang,object_state=obj,qpos=q,qvel=v,source_time=grid)
        accepted.append(dict(id=p.name,file=path.name,sha256=sha(path),frames=len(grid),seconds=float(grid[-1]-grid[0]),source_identity=e['identity'],source_frames_sha256=sha(p/'frames.csv'),source_model_sha256=modelhash))
    manifest=dict(schema='furry-dexweave-manip-reference-v1',accepted=accepted,rejected=rejected,joints=JOINTS,links=LINKS,independent_test_session=False,training='separate policy per successful reference',adaptations=['Recorded physical robot trajectory used directly; no human retargeting.','Unitree three-finger hands: 14 independent hand joints, six fingertips, 43 actions, actor244/critic437.','URDF exported from archived MuJoCo collision geometry and inertial frames; PhysX contact behavior must be validated separately.'])
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

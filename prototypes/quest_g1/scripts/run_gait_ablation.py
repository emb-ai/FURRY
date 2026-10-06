"""One-factor offline ablations. Generated experimental sources never enter APK.

Requires locally provisioned MuJoCo/ORT dependencies and an optional private
recording. Raw traces stay in ignored outputs/. Run from the project root.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def replace(text,old,new):
    assert text.count(old)==1, f'Expected one source anchor: {old[:100]}'
    return text.replace(old,new)

def sources(source_path=None):
    base=(source_path or ROOT/'android/native/meta_retarget.cpp').read_text().replace('headOrigin','translationOrigin')
    variants={'baseline':base}
    variants['old_up']=replace(replace(base,'left[1]=0;Normalize(left);','Normalize(left);mju_sub3(up,rest[1].position.data(),rest[0].position.data());Normalize(up);'),'mju_cross(forward,left,up);Normalize(forward);','mju_cross(forward,left,up);Normalize(forward);mju_cross(up,forward,left);')
    variants['no_scale']=replace(replace(base,'rootScale=robotPelvisHeight/humanPelvisHeight;','rootScale=1;'),'double armScale=robotReach/humanArm;','double armScale=1;')
    variants['upstream175']=replace(replace(base,'rootScale=robotPelvisHeight/humanPelvisHeight;','rootScale=.9*1.75/1.8;'),'double armScale=robotReach/humanArm;','double armScale=.8*1.75/1.8;')
    if 'translationOrigin=input.body.joints[0].position' in base:
        variants['pelvis_xy']=base
        variants['hmd_xy']=replace(replace(base,
            'translationOrigin=input.body.joints[0].position','translationOrigin=input.head.position'),
            'mju_sub3(pelvisDelta,input.body.joints[0].position.data(),translationOrigin.data());',
            'mju_sub3(pelvisDelta,input.head.position.data(),translationOrigin.data());')
    else:
        variants['pelvis_xy']=replace(replace(base,'translationOrigin=input.head.position','translationOrigin=input.body.joints[0].position'),'headDelta,input.head.position.data()','headDelta,input.body.joints[0].position.data()')
    # Undo just the ankle-landmark correction, retaining the fixed world up.
    variants['toe_scale']=replace(base,'double ankleHeight=.5*(gd->xpos[3*leftAnkle+2]+gd->xpos[3*rightAnkle+2]);','double ankleHeight=.5*(robot[6].position[2]+robot[7].position[2]);')
    # Anatomical proportions are not a rigid offset belonging to the foot.
    foot=replace(base,'namespace {','namespace {\nstd::array<std::array<double,3>,2> footProportion;')
    foot=replace(foot,'mju_mulMatTVec(offsets[i].position.data(),robotR,difference,3,3);','''if(i==6||i==7){
        int ankle=i==6?leftAnkle:rightAnkle;
        double rigid[3];mju_sub3(rigid,robot[i].position.data(),gd->xpos+3*ankle);
        for(int a=0;a<3;a++)footProportion[i-6][a]=difference[a]-rigid[a];
        mju_mulMatTVec(offsets[i].position.data(),robotR,rigid,3,3);
        }else mju_mulMatTVec(offsets[i].position.data(),robotR,difference,3,3);''')
    variants['foot_offset']=replace(foot,'for(int a=0;a<3;a++)targets[i].position[a]=root[a]+scales[i]*mapped[a]+offset[a];','''if(i==6||i==7){double correction[3];mju_rotVecQuat(correction,footProportion[i-6].data(),targets[0].quaternion.data());mju_addTo3(offset,correction);}
        for(int a=0;a<3;a++)targets[i].position[a]=root[a]+scales[i]*mapped[a]+offset[a];''')
    # Causal 40 ms low-pass on only root velocities; quantify tracking loss too.
    variants['velocity_filter']=replace(base,'mimic[0]=localVel[0];mimic[1]=localVel[1];mimic[2]=q[2];','''double alpha=dt>0 && dt<=.2?1-std::exp(-dt/.04):1;
    localVel[0]=mimic[0]+alpha*(localVel[0]-mimic[0]);
    localVel[1]=mimic[1]+alpha*(localVel[1]-mimic[1]);
    delta[5]=mimic[5]+alpha*(delta[5]-mimic[5]);
    mimic[0]=localVel[0];mimic[1]=localVel[1];mimic[2]=q[2];''')
    for name in ['meshes','no_drivetrain','no_self_contacts','blend2s','head_body_sync']:
        variants[name]=base
    return variants

def asset_variant(name,out):
    assets=ROOT/'android/assets'
    if name not in ('meshes','no_drivetrain','no_self_contacts'):return assets
    target=out/name/'assets';target.mkdir(parents=True,exist_ok=True)
    for f in assets.iterdir():
        if f.name=='scene.xml':continue
        if not (target/f.name).exists():(target/f.name).symlink_to(f)
    if name=='meshes':
        from g1_sim.scene import make_model
        _,xml=make_model('lab',hands=True)
        tree=ET.fromstring(xml)
    elif name=='no_self_contacts':
        tree=ET.parse(assets/'scene.xml').getroot()
        for geom in tree.find(".//body[@name='pelvis']").iter('geom'):
            if int(geom.get('contype','1')) or int(geom.get('conaffinity','1')):
                geom.set('contype','2');geom.set('conaffinity','1')
    else:
        tree=ET.parse(assets/'scene.xml').getroot()
        source=ET.parse(ROOT/'vendor/TWIST2/assets/g1/g1_sim2sim_29dof_with_hands.xml').getroot()
        joints={j.get('name'):j for j in source.iter('joint') if j.get('name')}
        for j in tree.iter('joint'):
            original=joints.get(j.get('name'))
            if original is None:continue
            for attr in ('armature','damping','frictionloss','actuatorfrcrange'):
                j.attrib.pop(attr,None)
                if attr in original.attrib:j.set(attr,original.get(attr))
    (target/'scene.xml').write_text(ET.tostring(tree,encoding='unicode'))
    return target

def run(name,source,args):
    folder=args.output/name;folder.mkdir(parents=True,exist_ok=True)
    (folder/'meta.cpp').write_text(source)
    replay=(ROOT/'android/native/replay_gmr_dynamics.cpp').read_text()
    if name=='blend2s':replay=replace(replay,'(r[3]-blendStart)/.5','(r[3]-blendStart)/2.')
    if name=='head_body_sync':
        replay=replace(replay,'std::ofstream out(argv[4]);','''// Interpolate the HMD to the body timestamp using already received
 // HMD samples only. Keep calibration HMD on the same time base.
 auto originals=poses;
 for(auto& entry:poses){auto& f=entry.second;auto hi=originals.upper_bound(entry.first);
   while(hi!=originals.begin()){--hi;if(hi->second.xr_time_ns<=f.body.time_ns)break;}
   auto lo=hi;if(hi!=originals.end() && hi->second.xr_time_ns<f.body.time_ns)++hi;
   if(hi!=originals.end() && hi->first<=entry.first && lo->second.xr_time_ns<=f.body.time_ns && hi->second.xr_time_ns>=f.body.time_ns){
     double a=hi==lo?0:double(f.body.time_ns-lo->second.xr_time_ns)/(hi->second.xr_time_ns-lo->second.xr_time_ns);
     for(int k=0;k<3;k++)f.head.position[k]=lo->second.head.position[k]+a*(hi->second.head.position[k]-lo->second.head.position[k]);
   }
 }
 std::ofstream out(argv[4]);''')
    (folder/'replay.cpp').write_text(replay)
    packages=ROOT/'.venv/lib/python3.12/site-packages'
    env=os.environ.copy();env['DYLD_LIBRARY_PATH']=f'{packages}/mujoco:{packages}/onnxruntime/capi'
    common=['clang++','-O2','-std=c++17','-Iandroid/native','-Ivendor/mujoco/include','-Ivendor/onnxruntime-android/headers',str(folder/'meta.cpp'),'android/native/gmr.cpp','android/native/simulation.cpp',str(packages/'mujoco/libmujoco.3.3.7.dylib'),str(packages/'onnxruntime/capi/libonnxruntime.1.23.2.dylib')]
    assets=asset_variant(name,args.output)
    jobs=[]
    if args.episode:jobs.append(('replay',str(folder/'replay.cpp'),[str(assets),str(args.episode),'retarget',str(folder/'replay.csv')]))
    if args.walk:jobs.append(('walk','android/native/ablation_motion.cpp',[str(assets),str(args.walk),'meta',str(folder/'walk.csv')]))
    jobs.append(('arms','android/native/ablation_motion.cpp',[str(assets),'arms','meta',str(folder/'arms.csv')]))
    for label,cpp,argv in jobs:
        binary=folder/label
        subprocess.run(common+[cpp,'-o',str(binary)],check=True,cwd=ROOT,env=env,capture_output=True)
        result=subprocess.run([str(binary),*argv],cwd=ROOT,env=env,text=True,capture_output=True,check=True)
        (folder/(label+'.log')).write_text(result.stdout+result.stderr)
        print(name,label,result.stdout.strip(),flush=True)
    return name

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline-source',type=Path);p.add_argument('--episode',type=Path);p.add_argument('--walk',type=Path);p.add_argument('--output',type=Path,default=Path('outputs/ablation'));p.add_argument('--variants',nargs='+');p.add_argument('--jobs',type=int,default=2);args=p.parse_args()
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    if args.episode:args.episode=args.episode.resolve()
    if args.walk:args.walk=args.walk.resolve()
    variants=sources(args.baseline_source);names=args.variants or list(variants)
    manifest=json.loads((args.output/'manifest.json').read_text()) if (args.output/'manifest.json').exists() else {}
    manifest.update({name:hashlib.sha256(variants[name].encode()).hexdigest() for name in names})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:list(pool.map(lambda name:run(name,variants[name],args),names))

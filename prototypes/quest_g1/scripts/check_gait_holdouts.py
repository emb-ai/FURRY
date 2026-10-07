"""Compare prebuilt source variants on untouched public walk clips, full duration."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import pickle
from pathlib import Path
from scipy.spatial.transform import Rotation
import subprocess
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from summarize_gait_ablation import summarize
ROOT=Path(__file__).resolve().parents[1]

def empty_scene_assets(source, target):
    """Copy a scene without LAB props, retaining original robot/floor assets."""
    source, target = source.resolve(), target.resolve()
    if source == target or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError('Generated empty-scene assets must be separate from source assets')
    tree = ET.parse(source/'scene.xml').getroot()
    parents = {child:parent for parent in tree.iter() for child in parent}
    for body in list(tree.iter('body')):
        if body.get('name') in {'table','cup','t_box','t_target'}:
            parents[body].remove(body)
    resources = [path for path in source.iterdir() if path.name != 'scene.xml']
    target.mkdir(parents=True, exist_ok=True)
    for resource in resources:
        link = target/resource.name
        if link.is_symlink():
            if link.resolve() == resource.resolve():
                continue
            link.unlink()
        elif link.exists():
            raise FileExistsError('Refusing to replace a non-symlink generated asset: '+str(link))
        link.symlink_to(resource.resolve(), target_is_directory=resource.is_dir())
    scene = target/'scene.xml'
    # Do not follow an old scene symlink back into production assets.
    if scene.is_symlink():
        scene.unlink()
    scene.write_text(ET.tostring(tree, encoding='unicode'))
    return target

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--prepare',action='store_true')
p.add_argument('--baseline',default='baseline',help='Baseline source variant directory name')
p.add_argument('--candidate',default='pelvis_xy',help='Candidate source variant directory name')
p.add_argument('--clips-dir',type=Path,help='Reuse normalized public clips from this directory')
p.add_argument('--assets',type=Path,default=ROOT/'android/assets',help='Fallback assets; per-variant assets take precedence')
p.add_argument('--empty-scene',action='store_true',help='Remove table, cup, T-object and target from copied scene assets')
p.add_argument('--results-dir',type=Path,help='Write rollouts/summary here instead of source directory/holdouts')
args=p.parse_args();folder=args.directory.resolve()
names=[args.baseline,args.candidate]
if len(set(names))!=2 or any(Path(name).name!=name or name in ('.','..') for name in names):
    p.error('Baseline and candidate must be distinct variant directory names')
clip_folder=args.clips_dir.resolve() if args.clips_dir else folder/'holdouts'
results_dir=args.results_dir.resolve() if args.results_dir else folder/'holdouts'
if args.empty_scene and (args.results_dir is None or results_dir == (folder/'holdouts').resolve()):
    p.error('--empty-scene requires a separate --results-dir to preserve LAB results')
assets={name:(folder/name/'assets' if (folder/name/'assets/scene.xml').exists() else args.assets.resolve()) for name in names}
if args.prepare:
    vendor=ROOT/'vendor/TWIST2'
    revision=subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip()
    if revision!='b06178f19a22f2138cbd31f60c6d494bc263f67d':raise SystemExit('Public motion preparation requires pinned TWIST2 source')
    output=clip_folder;output.mkdir(parents=True,exist_ok=True)
    # Only load the explicitly pinned repository's motion fixtures, never a
    # participant-supplied pickle. Private recordings use strict CSV schemas.
    for path in sorted((vendor/'assets/example_motions').glob('0807_yanjie_walk_*.pkl')):
        if path.stem.endswith('002'):continue
        with path.open('rb') as stream:clip=pickle.load(stream)
        pos=np.asarray(clip['root_pos'],dtype=float);rot=Rotation.from_quat(clip['root_rot'])
        yaw=rot[0].as_euler('xyz')[2];align=Rotation.from_euler('z',-yaw)
        pos=align.apply(pos-np.r_[pos[0,:2],0]);rot=align*rot
        q=np.c_[pos,rot.as_quat()[:,[3,0,1,2]],clip['dof_pos']]
        if q.shape[1]!=36 or not np.isfinite(q).all():raise ValueError('Invalid public motion')
        with (output/(path.stem+'.txt')).open('w') as stream:
            stream.write(str(clip['fps'])+'\n');np.savetxt(stream,q,fmt='%.17g')
clips=sorted(clip_folder.glob('*.txt'))
if not clips:raise SystemExit('No holdout clips; use --prepare with pinned TWIST2 assets')
if args.empty_scene:
    assets={name:empty_scene_assets(source,results_dir/'_assets'/name) for name,source in assets.items()}
packages=ROOT/'.venv/lib/python3.12/site-packages';env=os.environ.copy();env['DYLD_LIBRARY_PATH']=f'{packages}/mujoco:{packages}/onnxruntime/capi'
for name in names:
 binary=folder/name/'motion_full'
 cmd=['clang++','-O2','-std=c++17','-Iandroid/native','-Ivendor/mujoco/include','-Ivendor/onnxruntime-android/headers',str(folder/name/'meta.cpp'),'android/native/gmr.cpp','android/native/simulation.cpp','android/native/ablation_motion.cpp',str(packages/'mujoco/libmujoco.3.3.7.dylib'),str(packages/'onnxruntime/capi/libonnxruntime.1.23.2.dylib'),'-o',str(binary)]
 subprocess.run(cmd,cwd=ROOT,env=env,check=True)
def run(job):
 clip,name=job;out=results_dir/clip.stem/name;out.mkdir(parents=True,exist_ok=True)
 result=subprocess.run([str(folder/name/'motion_full'),str(assets[name]),str(clip),'meta',str(out/'walk.csv'),'full'],cwd=ROOT,env=env,check=True,text=True,capture_output=True)
 (out/'walk.log').write_text(result.stdout+result.stderr);print(clip.stem,name,result.stdout.strip(),flush=True)
with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,[(clip,name) for clip in clips for name in names]))
m=mujoco.MjModel.from_xml_path('android/assets/gmr_model.xml');result={}
for clip in clips:
 root=results_dir/clip.stem;entry={}
 for name in names:entry[name]=summarize(root/name/'walk.csv',root/args.baseline/'walk.csv',m)
 result[clip.stem]=entry
(results_dir/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
for clip,entry in result.items():
 print(clip,{name:{k:(round(r[k],4) if isinstance(r.get(k),(int,float)) else None) for k in ['falls','p95_tilt_deg','legs_velocity_above6Hz_rms_rad_s','wrist_common_target_rms_m','loaded_foot_slip_p95_m_s']} for name,r in entry.items()})

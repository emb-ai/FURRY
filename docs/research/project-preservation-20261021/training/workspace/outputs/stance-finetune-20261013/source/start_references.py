from pathlib import Path
import subprocess,os,json,shutil,hashlib
import mujoco,onnxruntime
R=Path.cwd(); O=R/'outputs/stance-finetune-20261013'; src=R/'.tools/furry-training/prototypes/quest_g1/training'
s=(src/'prepare.py').read_text().replace("sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))", "sys.path.insert(0,"+repr(str(src.parent/'scripts'))+")")
s=s.replace('str(src),str(dst)],check=True)',"str(src),str(dst),'clamp'],check=True)")
(O/'source/prepare_references.py').write_text(s);shutil.copy2(src/'motion_coordinates.py',O/'source/motion_coordinates.py')
processes=[]
for label,reduction in [('A',0),('B',.12)]:
 env=dict(os.environ,STANCE_REDUCTION_M=str(reduction),DYLD_LIBRARY_PATH=str(Path(mujoco.__file__).parent)+os.pathsep+str(Path(onnxruntime.__file__).parent/"capi"));cmd=[str(R/'.venv/bin/python'),str(O/'source/prepare_references.py'),str(R/'outputs/dataset-audit-20261008/episode/episode-1791467626074995'),str(O/(label+'-prepared')),'--assets',str(R/'outputs/quest-finetune-20261008/assets'),'--retarget-bin',str(O/'retarget')]
 log=open(O/(label+'-prepare.log'),'w');p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env); processes.append(dict(label=label,pid=p.pid,command=cmd,reduction_m=reduction))
(O/'reference-launch.json').write_text(json.dumps(processes,indent=2))
for p in processes:
 pid,status=os.waitpid(p['pid'],0);p['exit']=os.waitstatus_to_exitcode(status)
(O/'reference-exit.json').write_text(json.dumps(processes,indent=2))
assert all(p['exit']==0 for p in processes),processes

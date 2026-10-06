"""Export pinned GMR configuration/model for the native port (no mesh dependency)."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import mujoco
ROOT=Path(__file__).resolve().parents[1]
def export():
 out=ROOT/'android/assets';out.mkdir(exist_ok=True)
 source=ROOT/'vendor/GMR/assets/unitree_g1/g1_mocap_29dof.xml'
 m=mujoco.MjModel.from_xml_path(str(source));tree=ET.parse(source);root=tree.getroot()
 for parent in root.iter():
  for child in list(parent):
   if child.tag in ('geom','asset','keyframe','sensor','actuator'):parent.remove(child)
 for body in root.iter('body'):
  i=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,body.get('name',''))
  if i>0 and m.body_mass[i]>0 and body.find('inertial') is None:
   ET.SubElement(body,'inertial',pos=' '.join(map(str,m.body_ipos[i])),quat=' '.join(map(str,m.body_iquat[i])),mass=str(m.body_mass[i]),diaginertia=' '.join(map(str,m.body_inertia[i])))
 tree.write(out/'gmr_model.xml',encoding='unicode')
 model=mujoco.MjModel.from_xml_path(str(out/'gmr_model.xml'))
 assert (model.nq,model.nv)==(m.nq,m.nv)==(36,35)
 c=json.loads((ROOT/'android/config/gmr_xrobot_g1.json').read_text())
 names=list(c['human_scale_table'])
 with (out/'gmr_config.txt').open('w') as f:
  f.write(f"GMR1 {len(names)} {c['human_height_assumption']} {c['ground_height']}\n")
  for h in names:
   rows=[]
   for stage in (1,2):
    rows.append(next(( (b,v) for b,v in c[f'ik_match_table{stage}'].items() if v[0]==h)))
   assert rows[0][0]==rows[1][0] and rows[0][1][3:]==rows[1][1][3:]
   b,a=rows[0];v=rows[1][1]
   f.write(' '.join(map(str,[h,b,c['human_scale_table'][h],a[1],a[2],v[1],v[2],*a[3],*a[4]]))+'\n')
 return model
if __name__=='__main__':export()

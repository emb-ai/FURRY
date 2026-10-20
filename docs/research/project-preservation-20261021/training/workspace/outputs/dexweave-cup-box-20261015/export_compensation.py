"""Nominal hand-load gravity compensation geometry, from the archived model."""
import json
from pathlib import Path
import mujoco as mj
from dexcup.spec import JOINTS
p=Path(__file__).resolve().parent/'data/assets';m=mj.MjModel.from_binary_path(str(p/'reference.mjb'))
hands=[b for b in range(m.nbody) if '_hand_' in (mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,b) or '') or (mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,b) or '').endswith('wrist_yaw_link')]
joint_ids=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,n) for n in JOINTS[12:29]]
def descendant(child,parent):
    while child:
        if child==parent:return True
        child=int(m.body_parentid[child])
    return False
data=dict(hand_bodies=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,b) for b in hands],mass=m.body_mass[hands].tolist(),com=m.body_ipos[hands].tolist(),joint_bodies=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.jnt_bodyid[j])) for j in joint_ids],axis=m.jnt_axis[joint_ids].tolist(),mask=[[descendant(b,int(m.jnt_bodyid[j])) for b in hands] for j in joint_ids])
(p/'hand_compensation.json').write_text(json.dumps(data,indent=2))

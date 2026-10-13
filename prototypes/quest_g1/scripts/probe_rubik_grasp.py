"""Feasibility probe using G1 contacts and the ordinary TWIST/Grip interface.

Initial arm placement is a test fixture. During the measured grasp/lift only
commands are sent; neither the cube nor the robot's qpos are overwritten.
"""
import argparse
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from g1_sim.controller import Controller,standing_command,JOINTS
from g1_sim.scene import make_model
from scripts.rubik_native import NativePuzzle


def arm_ik(model,data,controller,side,position,rotation,guess):
    indices=np.arange(15,22) if side=="left" else np.arange(22,29)
    qa=controller.qadr[indices];jid=[model.joint(JOINTS[i]).id for i in indices]
    body=model.body(side+"_wrist_yaw_link").id
    def residual(q):
        data.qpos[qa]=q;mujoco.mj_kinematics(model,data)
        return np.r_[(data.xpos[body]-position)*10,
                     Rotation.from_matrix(data.xmat[body].reshape(3,3)@rotation.T).as_rotvec()]
    result=least_squares(residual,np.clip(guess,model.jnt_range[jid,0]+1e-5,model.jnt_range[jid,1]-1e-5),
        bounds=(model.jnt_range[jid,0]+1e-6,model.jnt_range[jid,1]-1e-6),max_nfev=60,ftol=1e-7,xtol=1e-7,gtol=1e-7)
    return result.x,np.linalg.norm(result.fun[:3])/10


def trial(offset,out,record=False,max_grip=.65,rotation=None,layer_offset=0):
    m,_=make_model("rubik");d=mujoco.MjData(m);c=Controller(m);c.initialize(d)
    r=NativePuzzle(m,d,out);r.reset();scratch=mujoco.MjData(m)
    center=d.xpos[m.body("rubik_core").id].copy()
    rotation=np.eye(3) if rotation is None else rotation
    position=center+[0,0,layer_offset]-rotation@np.array(offset)
    scratch.qpos[:]=d.qpos
    joints,error=arm_ik(m,scratch,c,"left",position,rotation,d.qpos[c.qadr[15:22]])
    d.qpos[c.qadr[15:22]]=joints;mujoco.mj_forward(m,d)
    initial_q=d.qpos.copy();initial_error=error;rows=[];max_height=0;contact_steps=0
    command=standing_command();command[21:28]=joints
    for k in range(3500):
        t=k*.001;grip=np.clip((t-.2)/.8,0,1)*max_grip
        lift=np.clip((t-1.3)/1.0,0,1);lift=lift*lift*(3-2*lift)*.10
        if k%20==0:
            scratch.qpos[:]=d.qpos
            joints,error=arm_ik(m,scratch,c,"left",position+[0,0,lift],rotation,joints)
            command[21:28]=joints
        r.begin(grip,0);c.step(d,command,grip,direct_arms=True,right_grip=0);r.finish()
        current=d.xpos[m.body("rubik_core").id]
        mask=r.contacts(0);contact_steps+=bool(mask);max_height=max(max_height,current[2])
        if k%10==0:rows.append([d.time,*current,*d.xpos[m.body("left_wrist_yaw_link").id],mask,d.qpos[2],*d.userdata[:5]])
        if not np.isfinite(d.qpos).all() or d.qpos[2]<.35:break
    result={"offset":offset,"max_grip":max_grip,"rotation":rotation.tolist(),"layer_offset":layer_offset,"initial_cube_center":center.tolist(),"lift_m":float(max_height-center[2]),"initial_arm_ik_error_m":float(initial_error),"max_cube_center_z_m":float(max_height),
            "final_cube_center":d.xpos[m.body("rubik_core").id].tolist(),"contact_steps":contact_steps,
            "minimum_robot_z_m":float(min(x[8] for x in rows)),"warnings":sum(w.number for w in d.warning)}
    if record:np.savez(out/"grasp-trace.npz",rows=np.array(rows),initial_qpos=initial_q)
    r.close();return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("output",type=Path);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for x,y in [(.12,-.06),(.12,-.05),(.13,-.06),(.11,-.055),(.13,-.05),(.14,-.055)]:
        result=trial([x,y,0],args.output);results.append(result);print(json.dumps(result),flush=True)
        (args.output/"grasp-search.json").write_text(json.dumps(results,indent=2)+"\n")
    candidates=[r for r in results if r["final_cube_center"][2]>r["initial_cube_center"][2]+.04 and r["contact_steps"]>1000 and r["warnings"]==0]
    if not candidates:raise RuntimeError("No supported physical lift; inspect grasp-search.json")
    best=max(candidates,key=lambda r:r["final_cube_center"][2]);trial(best["offset"],args.output,True)


if __name__=="__main__":main()

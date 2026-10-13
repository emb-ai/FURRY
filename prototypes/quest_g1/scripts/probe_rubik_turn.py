"""Unaccepted controller-command feasibility probe: ordinary Grip, free cube, G1 hands.

Arm IK produces controller-equivalent targets. Puzzle motion comes exclusively
from MuJoCo contacts, the passive hinge and the magnetic torque in RubikTask.
Raw turn counts are diagnostics; inspect physical cubie poses before accepting.
"""
import argparse
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from g1_sim.scene import make_model
from g1_sim.controller import Controller,standing_command
from scripts.rubik_native import NativePuzzle
from scripts.probe_rubik_grasp import arm_ik


def run(out):
    m,xml=make_model('rubik');d=mujoco.MjData(m);c=Controller(m);c.initialize(d)
    task=NativePuzzle(m,d,out);task.reset();scratch=mujoco.MjData(m);scratch.qpos[:]=d.qpos
    leftRotation=np.eye(3)
    leftPos=np.array([.13,.05,.931]);left,error=arm_ik(m,scratch,c,'left',leftPos,leftRotation,d.qpos[c.qadr[15:22]])
    d.qpos[c.qadr[15:22]]=left;mujoco.mj_forward(m,d)
    command=standing_command();command[21:28]=left;right=d.qpos[c.qadr[22:29]].copy()
    rows=[];positions=[];velocities=[];states=[];peak=0;hand_contacts=[0,0];rightError=0
    for k in range(14500):
        t=k*.001
        leftGrip=np.clip((t-.2)/.8,0,1)*.65
        lift=np.clip((t-1.3)/1.,0,1);lift=lift*lift*(3-2*lift)*.10
        rightGrip=np.clip((t-3.2)/.6,0,1)*.40
        angle=-np.clip((t-6.8)/2.5,0,1)*np.pi/2
        if k%20==0:
            scratch.qpos[:]=d.qpos
            left,_=arm_ik(m,scratch,c,'left',leftPos+[0,0,lift],leftRotation,left)
            command[21:28]=left
            if t>3.6:
                center=d.xpos[m.body('rubik_core').id].copy();cubeRotation=d.xmat[m.body('rubik_core').id].reshape(3,3)
                orientation=cubeRotation@Rotation.from_euler('z',angle).as_matrix()
                # Ordinary right Grip=.4 locates its index at approximately
                # (168,69,-28.5) mm in the wrist frame. Aim that physical
                # fingertip at an upper cubie and sweep it tangentially.
                target=center+orientation@np.array([.020-.168,-.025-.069,.018+.0285])
                target[2]+=(1-np.clip((t-3.6)/1.8,0,1))*.08
                right,rightError=arm_ik(m,scratch,c,'right',target,orientation,right)
                command[28:35]=right
        task.begin(leftGrip,rightGrip)
        c.step(d,command,leftGrip,direct_arms=True,right_grip=rightGrip)
        task.finish()
        for hand in (0,1):hand_contacts[hand]+=bool(task.contacts(hand))
        peak=max(peak,float(d.xpos[m.body('rubik_core').id,2]))
        if k%10==0:
            rows.append([d.time,*d.xpos[m.body('rubik_core').id],*d.userdata[:5],leftGrip,rightGrip,angle,task.contacts(0),task.contacts(1),
                         *d.xpos[m.body('right_wrist_yaw_link').id],rightError if t>3.6 else 0])
            positions.append(d.qpos.copy());velocities.append(d.qvel.copy());states.append(d.userdata.copy())
    result={'max_cube_z_m':peak,'final_cube':d.xpos[m.body('rubik_core').id].tolist(),'turns':int(d.userdata[2]),
            'last_face':int(d.userdata[3]),'left_contact_steps':hand_contacts[0],'right_contact_steps':hand_contacts[1],
            'warnings':sum(w.number for w in d.warning)}
    np.savez(out/'turn-probe.npz',rows=rows,qpos=positions,qvel=velocities,userdata=states,scene_xml=xml)
    (out/'turn-probe.json').write_text(json.dumps(result,indent=2)+'\n');task.close();print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args();args.output.mkdir(exist_ok=True,parents=True);run(args.output)

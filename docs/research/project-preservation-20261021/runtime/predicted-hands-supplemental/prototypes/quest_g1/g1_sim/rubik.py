"""Physical 2x2 corner puzzle: free core, eight free cubies, switchable layers.

Welds model the internal retaining mechanism, never attach the cube to a hand.
The active hinge is passive: only hand-contact forces can drive a face turn.
"""
import math
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .cup_lab import el, decoration

FACES = ("R", "L", "B", "F", "U", "D")
AXES = np.array([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)],dtype=float)
CENTERS = np.array([(x,y,z) for x in (-.016,.016) for y in (-.016,.016) for z in (-.016,.016)])
SCRAMBLE = ((0,1),(4,-1),(3,1),(0,-1))
STATE_SIZE = 64


def rotate_state(positions,quaternions,face,sign):
    axis=AXES[face];rotation=np.r_[math.cos(math.pi/4),axis*sign*math.sin(math.pi/4)]
    matrix=np.zeros(9);mujoco.mju_quat2Mat(matrix,rotation)
    for i in np.flatnonzero(positions@axis>.008):
        positions[i]=np.round(matrix.reshape(3,3)@positions[i],12)
        q=np.empty(4);mujoco.mju_mulQuat(q,rotation,quaternions[i]);quaternions[i]=q


def initial_state(scramble=SCRAMBLE):
    p=CENTERS.copy();q=np.tile([1.,0,0,0],(8,1))
    for face,sign in scramble:rotate_state(p,q,face,sign)
    return p,q


def add_puzzle(root):
    world=root.findall("worldbody")[-1];asset=root.find("asset")
    # Reuse the cup workcell, then replace its object and placement inlay.
    world.remove(root.find(".//body[@name='cup']"))
    table=root.find(".//body[@name='table']")
    table.set("pos",".50 0 0")
    root.find("custom/numeric[@name='cup_lab_goal']").set("data",".50 -.25 .825 .055")
    for geom in list(table.findall("geom")):
        if geom.get("name", "").startswith("cup_place_"):table.remove(geom)
        elif geom.get("name")=="tabletop":geom.set("pos","0 0 .800")
        elif geom.get("name", "").startswith("lab_bench_leg_"):
            p=list(map(float,geom.get("pos").split()));p[2]=.3875;geom.set("pos"," ".join(map(str,p)))
            geom.set("size",".024 .024 .3875")
        elif geom.get("name", "").startswith("lab_bench_apron_"):
            p=list(map(float,geom.get("pos").split()));p[2]+=.075;geom.set("pos"," ".join(map(str,p)))
    el(asset,"material",name="rubik_plastic",rgba=".035 .045 .05 1",specular=.28,shininess=.35)
    colors=((.82,.065,.04,1),(.98,.34,.035,1),(.04,.20,.78,1),(.045,.62,.20,1),(.94,.94,.85,1),(.98,.77,.025,1))
    for name,color in zip(FACES,colors):el(asset,"material",name="rubik_"+name,rgba=color,specular=.3,shininess=.35)
    # A dark rubber starting zone. It is a flush visual inlay, not an obstacle.
    decoration(table,"rubik_start_zone",pos=(-.25,0,.8255),size=(.075,.085,.0004),material="lab_metal")
    # A narrow presentation stand leaves clearance for G1's long fingertips.
    # The cube simply rests on it; there is no constraint or grasp attachment.
    el(table,"geom",name="rubik_stand",type="cylinder",pos=(-.25,0,.8625),size=(.014,.0375),
       material="lab_metal",friction="1 .005 .0001")
    pos=np.array([.25,0,.934]);p,q=initial_state()
    core=el(world,"body",name="rubik_core",pos=pos.tolist())
    el(core,"freejoint",name="rubik_free")
    el(core,"inertial",pos=(0,0,0),mass=.032,diaginertia=(.000009,.000009,.000009))
    for face,axis in zip(FACES,AXES):
        layer=el(core,"body",name="rubik_layer_"+face)
        el(layer,"inertial",pos=(0,0,0),mass=.001,diaginertia=(.0000002,)*3)
        el(layer,"joint",name="rubik_hinge_"+face,type="hinge",axis=axis.tolist(),limited="false",actuatorfrclimited="false",actuatorfrcrange="-1 1",armature=.00001,damping=.0008,frictionloss=0)
    equality=root.find("equality")
    if equality is None:equality=el(root,"equality")
    for i,(cp,cq) in enumerate(zip(p,q)):
        cubie=el(world,"body",name=f"rubik_cubie_{i}",pos=(pos+cp).tolist(),quat=cq.tolist())
        el(cubie,"freejoint",name=f"rubik_cubie_free_{i}")
        el(cubie,"geom",name=f"rubik_block_{i}",type="box",size=(.0154,)*3,mass=.0105,
           material="rubik_plastic",friction="1.1 .005 .0001",condim=4,contype=16,conaffinity=15,
           solref=".004 1",solimp=".98 .995 .001")
        for axis in range(3):
            sign=1 if CENTERS[i,axis]>0 else -1
            s=[.0127,.0127,.0127];s[axis]=.00045
            location=[0.,0.,0.];location[axis]=sign*.01585
            color=FACES[axis*2+(sign<0)]
            decoration(cubie,f"rubik_sticker_{i}_{axis}",pos=location,size=s,material="rubik_"+color)
        el(equality,"weld",name=f"rubik_core_weld_{i}",body1="rubik_core",body2=f"rubik_cubie_{i}",
           solref=".002 1",solimp=".99 .999 .001",torquescale=.2)
        for face in FACES:
            el(equality,"weld",name=f"rubik_layer_weld_{face}_{i}",body1="rubik_layer_"+face,body2=f"rubik_cubie_{i}",
               active="false",solref=".002 1",solimp=".99 .999 .001",torquescale=.2)
    size=root.find("size")
    if size is None:size=el(root,"size")
    size.set("nuserdata",str(STATE_SIZE))
    custom=root.find("custom")
    el(custom,"numeric",name="rubik_initial",data=np.c_[p,q].ravel().tolist())
    el(custom,"numeric",name="rubik_magnet",data=(.24,15*math.pi/180,.00012))
    el(custom,"text",name="rubik_scramble",data="R U' F R'")
    el(custom,"text",name="rubik_model",data="2x2-magnetic-contact-v1")

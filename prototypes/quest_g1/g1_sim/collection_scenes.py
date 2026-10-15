"""Authored rigid-body adaptations of TWIST2 manipulation tasks.

WB-Dex follows paper section IV-C (shelf -> floor box). Dimensions, masses,
randomization and scoring are FURRY parameters, not unpublished author values.
Bimanual and basket are simplified collection tasks inspired by IV-A/B; the
basket includes a pushable door; the towel task is not approximated
with rigid objects. No meshes/code from another project are copied here.
"""
import math
from pathlib import Path
from .cup_lab import el, lathe, GOAL

PAPER = 'https://arxiv.org/html/2511.02832v1#S4'
TASKS = {
    'cup': ('Чашка на круг', 'tabletop_place'),
    'wb_dex': ('Чашка с полки в коробку', 'shelf_to_floor_box'),
    'bimanual': ('Два предмета в два лотка', 'two_object_sort'),
    'basket': ('Корзина через дверь', 'basket_through_door'),
}


def box(parent, name, pos, size, color, mass=None, visual=False):
    args = dict(name=name, type='box', pos=pos, size=size, rgba=color)
    if mass is not None:
        args['mass'] = mass
    if visual:
        args.update(contype=0, conaffinity=0, mass=0)
    return el(parent, 'geom', **args)


def tray(world, name, pos, half=(.19,.20), height=.13):
    b=el(world,'body',name=name,pos=pos)
    box(b,name+'_bottom',(0,0,.009),(*half,.009),(.25,.5,.42,1))
    for axis in range(2):
        for sign in (-1,1):
            q=[0,0,height/2];q[axis]=sign*half[axis]
            dims=[*half,height/2];dims[axis]=.008
            box(b,f'{name}_wall{axis}_{sign}',q,dims,(.34,.59,.50,1))
    return b


def shelf(world, name, pos, half, top):
    b=el(world,'body',name=name,pos=pos)
    box(b,name+'_top',(0,0,top-.018),(*half,.018),(.60,.44,.28,1))
    for x in (-half[0]+.035,half[0]-.035):
        for y in (-half[1]+.035,half[1]-.035):
            box(b,f'{name}_leg{x}_{y}',(x,y,(top-.036)/2),(.024,.024,(top-.036)/2),(.19,.23,.25,1))
    return b


def compact_robot_visuals(root):
    for name in ('torso_link','head_link'):
        for mesh in root.find('asset').findall('mesh'):
            if (mesh.get('name')==name or Path(mesh.get('file','')).stem==name) and mesh.get('file','').endswith('.obj'):
                mesh.set('name',name)
                mesh.set('file',str(Path(__file__).resolve().parents[1]/'assets/g1/visual/cup_lab'/('cup-lab-'+name+'.obj')))


def add_collection_scene(root, scene):
    if scene not in TASKS:
        return
    compact_robot_visuals(root)
    world=root.findall('worldbody')[-1]
    custom=root.find('custom')
    if custom is None: custom=el(root,'custom')
    size=root.find('size')
    if size is None: size=el(root,'size')
    size.set('nuserdata','8')
    # type, horizon seconds; all temporal state lives in userdata.
    el(custom,'numeric',name='collection_task',data=(list(TASKS).index(scene)+1,90))
    if scene=='cup':
        # Keep the cup base supported within the standing manipulation workspace.
        # The handle faces away from the robot (+X), with a small yaw variation,
        # so the power grasp closes on the body instead of hooking the loop.
        el(custom,'numeric',name='collection_start_0',data=(.40,.49,-.28,.28,.754,.10))
        el(custom,'numeric',name='collection_goal_0',data=(*GOAL[:3],GOAL[3],GOAL[3],.02))
        # Equal mixture of short and long transfers. The target is an episode
        # state (mocap), so one unchanged model can represent every layout.
        el(custom,'numeric',name='collection_cup_goal_bounds',data=(.40,.49,-.28,.28,.15,.30,.32,.55))
        marker=el(world,'body',name='cup_goal_marker',mocap='true',pos=GOAL[:3])
        table=root.find(".//body[@name='table']")
        for geom in list(table.findall('geom')):
            if geom.get('name','').startswith('cup_place_'):
                table.remove(geom)
                xyz=list(map(float,geom.get('pos').split()))
                xyz=[xyz[0]+.65-GOAL[0],xyz[1]-GOAL[1],xyz[2]-GOAL[2]]
                geom.set('pos',' '.join(map(str,xyz)));marker.append(geom)
        return
    if scene=='wb_dex':
        # Floor placement requires deep squatting; standing-height failure is invalid here.
        el(custom,'numeric',name='collection_fall',data=(.25,math.cos(math.pi/4)))
        shelf(world,'low_shelf',(.70,0,0),(.24,.43),.55)
        cup=el(world,'body',name='task_object_0',pos=(.60,0,.554))
        el(cup,'freejoint',name='task_object_0_free')
        # Light hollow cup, .060 kg total; no artificial grasp attachment.
        el(cup,'geom',name='task_cup_bottom',type='cylinder',size=(.035,.003),pos=(0,0,.003),mass=.012,rgba=(.88,.73,.34,1))
        for i in range(16):
            a=i*2*math.pi/16
            box(cup,f'task_cup_wall_{i}',(.033*math.cos(a),.033*math.sin(a),.048),(.003,.007,.042),(.88,.73,.34,1),.003).set('euler',f'0 0 {a}')
        tray(world,'floor_box',(.58,-.64,0),(.20,.22),.18)
        el(custom,'numeric',name='collection_start_0',data=(.55,.83,-.30,.30,.554,math.pi))
        el(custom,'numeric',name='collection_goal_0',data=(.58,-.64,.018,.15,.17,.03))
    elif scene=='bimanual':
        el(custom,'numeric',name='collection_bimanual_hands',data=(1,))
        shelf(world,'sorting_table',(.60,0,0),(.32,.64),.75)
        for i,sign in enumerate((1,-1)):
            obj=el(world,'body',name=f'task_object_{i}',pos=(.42,sign*.17,.799))
            el(obj,'freejoint',name=f'task_object_{i}_free')
            color=(.90,.44,.18,1) if i==0 else (.19,.57,.85,1)
            # 50 x 50 x 90 mm spans both finger pads without forcing a wide fist.
            block=box(obj,f'sort_block_{i}',(0,0,0),(.025,.025,.045),color,.085)
            block.set('solref','.004 1');block.set('solimp','.98 .995 .001');block.set('condim','4')
            receptacle=tray(world,f'sort_tray_{i}',(.46,sign*.40,.75),(.12,.13),.035)
            for geom in receptacle.findall('geom'):geom.set('rgba',' '.join(map(str,color)))
            el(custom,'numeric',name=f'collection_start_{i}',data=(.38,.46,sign*.16-.055,sign*.16+.055,.799,.7))
            # Sorting accepts either resting orientation: centers are .793/.813 m.
            el(custom,'numeric',name=f'collection_goal_{i}',data=(.46,sign*.40,.803,.075,.085,.018))
    else:
        # Hinged door separates pickup and placement. It opens by contact;
        # no animation or hidden motor pulls the robot/basket through it.
        el(custom,'numeric',name='collection_door',data=(1.65,.60))
        el(custom,'numeric',name='collection_basket_sides',data=(1,))
        frame=el(world,'body',name='basket_door_frame')
        for sign in (-1,1):
            box(frame,f'door_post{sign}',(1.65,sign*.64,1.02),(.055,.04,1.02),(.26,.30,.31,1))
            box(frame,f'door_wall{sign}',(1.65,sign*1.39,1.),(.04,.71,1.),(.55,.60,.61,1))
        box(frame,'door_lintel',(1.65,0,2.06),(.055,.68,.04),(.26,.30,.31,1))
        door=el(world,'body',name='basket_door',pos=(1.65,.60,0))
        el(door,'joint',name='basket_door_hinge',type='hinge',axis=(0,0,1),actuatorfrclimited='false',limited='true',range=(0,1.65),damping=.7,frictionloss=.15,armature=.01)
        box(door,'door_panel',(0,-.60,1.0),(.025,.59,.99),(.60,.43,.27,1),3.)
        box(door,'door_push_plate',(-.034,-1.04,1.),(.009,.065,.12),(.25,.30,.32,1),.05)
        shelf(world,'basket_destination',(3.25,0,0),(.32,.55),.64)
        basket=el(world,'body',name='task_object_0',pos=(.57,.53,.005))
        el(basket,'freejoint',name='task_object_0_free')
        box(basket,'basket_base',(0,0,.01),(.16,.19,.01),(.68,.48,.24,1),.16)
        for sign in (-1,1):
            box(basket,f'basket_end{sign}',(sign*.156,0,.075),(.008,.19,.065),(.76,.57,.31,1),.04)
            box(basket,f'basket_side{sign}',(0,sign*.186,.075),(.15,.008,.065),(.76,.57,.31,1),.04)
            # Real open handles with room for the three fingers.
            for x in (-.09,.09):
                el(basket,'geom',name=f'handle_post{sign}_{x}',type='capsule',fromto=(x,sign*.205,.10,x,sign*.205,.22),size=.009,mass=.01,rgba=(.16,.21,.23,1))
            el(basket,'geom',name=f'handle_bar{sign}',type='capsule',fromto=(-.09,sign*.205,.22,.09,sign*.205,.22),size=.011,mass=.02,rgba=(.16,.21,.23,1))
        box(world,'basket_goal',(3.25,0,.642),(.22,.25,.002),(.20,.65,.47,.7),visual=True)
        el(custom,'numeric',name='collection_start_0',data=(.53,.61,.49,.57,.005,.12))
        el(custom,'numeric',name='collection_goal_0',data=(3.25,0,.64,.09,.08,.022))


def add_feeder(root,world):
    """Purpose-built table-tennis robot with wheels, hopper and aiming head.

    Head +Z points out of the muzzle. Native trajectory solver drives its mocap
    orientation before firing; geometry is visual and cannot bat the ball.
    """
    compact_robot_visuals(root)
    b=el(world,'body',name='tennis_feeder',pos=(3.39,0,0))
    box(b,'feeder_chassis',(0,0,.12),(.24,.25,.08),(.13,.17,.21,1))
    for x in (-.15,.15):
        for y in (-.25,.25):
            el(b,'geom',name=f'feeder_wheel{x}_{y}',type='cylinder',pos=(x,y,.10),size=(.095,.035),euler=(math.pi/2,0,0),rgba=(.045,.055,.07,1))
    box(b,'feeder_column',(.07,0,.65),(.085,.09,.46),(.29,.36,.40,1))
    box(b,'feeder_front',(-.027,0,.72),(.013,.10,.18),(.91,.51,.16,1))
    box(b,'feeder_screen',(-.044,0,.80),(.005,.067,.035),(.12,.84,.82,1),visual=True)
    # Yoke and broad pivot visibly support the aiming head.
    box(b,'feeder_yoke_bridge',(-.09,0,1.14),(.18,.16,.035),(.22,.29,.32,1))
    for sign in (-1,1):
        box(b,f'feeder_yoke{sign}',(-.19,sign*.14,1.24),(.035,.025,.10),(.28,.35,.38,1))
        el(b,'geom',name=f'feeder_pivot{sign}',type='sphere',pos=(-.19,sign*.12,1.29),size=.055,rgba=(.30,.37,.39,1),contype=0,conaffinity=0,mass=0)
    # Open faceted hopper and visible reserve balls.
    box(b,'hopper_base',(.11,0,1.39),(.17,.18,.02),(.15,.20,.24,1))
    for sign in (-1,1):
        box(b,f'hopper_side{sign}',(.11,sign*.18,1.51),(.17,.012,.12),(.33,.42,.46,.8))
        box(b,f'hopper_end{sign}',(.11+sign*.17,0,1.51),(.012,.18,.12),(.33,.42,.46,.8))
    for i in range(9):
        el(b,'geom',name=f'hopper_ball{i}',type='sphere',size=.02,pos=(.0+(i%3)*.065,-.09+(i//3)*.065,1.43+(i%2)*.015),rgba=(1,.88,.42,1),contype=0,conaffinity=0,mass=0)
    head=el(world,'body',name='tennis_aim_head',mocap='true',pos=(3,0,1.32),quat=(.70710678,0,-.70710678,0))
    def visual(name,kind='box',**args):
        return el(head,'geom',name=name,type=kind,contype=0,conaffinity=0,mass=0,**args)
    visual('feeder_head_housing',size=(.12,.11,.11),pos=(0,0,-.20),rgba=(.86,.47,.14,1))
    # Open barrel ring; inner dark surface and bright longitudinal sight line.
    asset=root.find('asset')
    lathe(asset,'feeder_barrel_mesh',[(.052,-.17),(.052,0),(.029,0),(.029,-.17),(.052,-.17)],32)
    visual('feeder_barrel','mesh',mesh='feeder_barrel_mesh',rgba=(.15,.20,.24,1))
    visual('feeder_sight',size=(.008,.008,.11),pos=(0,.057,-.10),rgba=(.12,.91,.82,1))
    for sign in (-1,1):
        visual(f'feeder_motor{sign}','cylinder',size=(.075,.025),pos=(sign*.13,0,-.20),euler=(0,math.pi/2,0),rgba=(.23,.29,.33,1))
    # Physical housing/motors and an open ring of barrel colliders. The bore
    # stays open so the newly served ball is not born inside a convex hull.
    for geom in head.findall('geom'):
        if geom.get('name')=='feeder_head_housing' or geom.get('name','').startswith('feeder_motor'):
            geom.set('contype','16');geom.set('conaffinity','31')
    for i in range(16):
        a=2*math.pi*i/16;x=.043*math.cos(a);y=.043*math.sin(a)
        el(head,'geom',name=f'barrel_collision_{i}',type='capsule',fromto=(x,y,-.17,x,y,0),size=.008,
           contype=16,conaffinity=31,mass=0,group=3,rgba=(0,0,0,0))
    # Two bright eyes make the head direction legible at room scale.
    for sign in (-1,1):
        visual(f'feeder_eye{sign}','sphere',size=.018,pos=(sign*.080,.070,-.079),rgba=(.20,.95,.88,1))

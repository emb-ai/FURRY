"""Build scenes with source articulation and independently authored geometry."""
import math
import xml.etree.ElementTree as ET
import mujoco
from .controller import UPSTREAM
from .geometry import apply_hybrid, configure_props
from .cup_lab import add_workcell
from .rubik import add_puzzle
from .collection_scenes import add_collection_scene, add_feeder


def add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def make_model(scene="lab", hands=True, geometry="hybrid"):
    if geometry not in ("hybrid", "source", "collision_only"):
        raise ValueError(f"Unknown geometry variant: {geometry}")
    source = UPSTREAM / "assets/g1" / ("g1_sim2sim_29dof_with_hands.xml" if hands else "g1_sim2sim_29dof.xml")
    root = ET.parse(source).getroot()
    if hands:
        # The upstream hand variant omits several drivetrain inertias present in
        # the policy's sim2sim model. Restore these for the same 29 body joints.
        reference = ET.parse(source.parent / "g1_sim2sim_29dof.xml").getroot()
        body_joints = {j.get("name"): j for j in reference.iter("joint") if j.get("name")}
        for joint in root.iter("joint"):
            if joint.get("name") in body_joints:
                ref = body_joints[joint.get("name")]
                for attr in ("armature", "damping", "frictionloss", "actuatorfrcrange"):
                    if attr in ref.attrib:
                        joint.set(attr, ref.get(attr))
    # PGS saturates 50 sweeps on dense finger/cup contacts. Newton solves the
    # same contact model to the existing tolerance without changing timestep.
    root.find("option").set("solver", "Newton")
    if scene in ("cup", "bimanual", "rubik", "wb_dex"):
        # A cylinder bottom against a box needs a contact patch. Single-point
        # CCD makes a stationary cup rock and repeatedly fail the settle gate.
        # The sorting blocks also need patches against the flat finger pads.
        flag = root.find("option/flag")
        if flag is None:
            flag = ET.SubElement(root.find("option"), "flag")
        flag.set("multiccd", "enable")
        # Soft friction otherwise creeps under a static power grasp and drops
        # the cup during a one-minute hold. Standard friction-only NoSlip
        # sweeps suppress that regularization drift without attaching objects
        # or raising motor force/friction coefficients.
        if scene != "wb_dex":
            root.find("option").set("noslip_iterations", "2")
    if hands and scene in ("cup", "rubik", "wb_dex", "bimanual", "basket"):
        # Grip is an angle target: a blocked finger must not keep squeezing
        # toward a closed fist with the upstream 2.45 Nm joint limit.
        # Cap motor effort independently of closure angle. Joint damping is
        # implicit under Euler; explicit actuator velocity feedback is unstable
        # for the tiny distal inertias at the production 1 ms timestep.
        for actuator in root.findall("actuator/position"):
            if "hand_" in actuator.get("name", ""):
                actuator.set("forcelimited", "true")
                actuator.set("forcerange", "-.35 .35")
        for joint in root.iter("joint"):
            if "hand_" in joint.get("name", ""):
                joint.set("damping", ".03")
    root.find("compiler").set("meshdir", str(source.parent / "meshes"))
    if hands and geometry != "source":
        apply_hybrid(root, visuals=geometry == "hybrid")
    # The supplied keyframe is for the original qpos layout, not extra free props.
    for key in root.findall("keyframe"):
        root.remove(key)
    root.set("model", "Unitree G1 | TWIST2 | " + scene)
    visual = root.find("visual")
    if visual is None:
        visual = add(root, "visual")
    glob = visual.find("global")
    if glob is None:
        glob = add(visual, "global")
    glob.set("offwidth", "1280")
    glob.set("offheight", "800")
    for texture in root.iter("texture"):
        if texture.get("type") == "skybox":
            texture.set("rgb1", ".14 .19 .26")
            texture.set("rgb2", ".38 .46 .54")
        elif texture.get("name") == "groundplane":
            texture.set("rgb1", ".23 .28 .32")
            texture.set("rgb2", ".29 .34 .38")
            texture.set("markrgb", ".36 .41 .45")
    world = root.findall("worldbody")[-1]
    add(world, "light", pos="1 -3 4", dir="-.2 .5 -1", diffuse=".7 .7 .7")
    # Camera looks along robot +X; MuJoCo camera looks along local -Z.
    torso = root.find(".//body[@name='torso_link']")
    add(torso, "camera", name="ego", pos=".08 0 .43", xyaxes="0 -1 0 0 0 1", fovy="80")
    if scene in ("cup", "lab", "rubik"):
        x, y = (.65, 0.) if scene in ("cup", "rubik") else (.8, -.8)
        table = add(world, "body", name="table", pos=f"{x} {y} 0")
        add(table, "geom", name="tabletop", type="box", size=".32 .45 .025", pos="0 0 .725", rgba=".42 .26 .15 1", friction=".8 .005 .0001")
        for dx in (-.26, .26):
            for dy in (-.39, .39):
                add(table, "geom", type="box", size=".025 .025 .35", pos=f"{dx} {dy} .35", rgba=".15 .18 .2 1")
        cup = add(world, "body", name="cup", pos=f"{x-.12} {y} .754")
        add(cup, "freejoint", name="cup_free")
        add(cup, "geom", name="cup_bottom", type="cylinder", size=".04 .004", pos="0 0 .004", mass=".06", rgba=".1 .65 .8 1", friction="1 .005 .0001")
        for i in range(16):
            angle = i*2*math.pi/16
            add(cup, "geom", name=f"cup_wall_{i}", type="box", size=".004 .008 .047", pos=f"{.039*math.cos(angle)} {.039*math.sin(angle)} .051", euler=f"0 0 {angle}", mass=".009", rgba=".1 .65 .8 1", friction="1 .005 .0001")
        for i, (a, b) in enumerate([((.042,0,.08),(.076,0,.08)), ((.076,0,.08),(.076,0,.027)), ((.076,0,.027),(.042,0,.027))]):
            add(cup, "geom", name=f"cup_handle_{i}", type="capsule", size=".006", fromto=" ".join(map(str, (*a,*b))), mass=".012", rgba=".1 .65 .8 1")
    if scene in ("push_t", "lab"):
        x, y = (.65, 0.) if scene == "push_t" else (.9, .85)
        box = add(world, "body", name="t_box", pos=f"{x} {y} .101")
        add(box, "freejoint", name="t_box_free")
        for name, pos, size, mass in [("bar", ".12 0 0", ".09 .3 .1", 1.8), ("stem", "-.12 0 0", ".15 .09 .1", 1.2)]:
            add(box, "geom", name=f"t_{name}", type="box", pos=pos, size=size, mass=mass, rgba=".22 .7 .42 1", friction=".5 .005 .0001")
        target = add(world, "body", name="t_target", pos=f"{x+.9} {y} .002")
        for pos, size in [(".12 0 0", ".09 .3 .002"), ("-.12 0 0", ".15 .09 .002")]:
            add(target, "geom", type="box", pos=pos, size=size, contype="0", conaffinity="0", rgba=".15 .7 1 .45")
    if scene == "tennis":
        # +X is opponent side; near edge is reachable from standing G1.
        table = add(world, "body", name="tennis_table", pos="1.77 0 0")
        add(table, "geom", name="tennis_top", type="box", size="1.37 .7625 .015", pos="0 0 .745", rgba=".035 .22 .4 1")
        for dx in (-1.12, 1.12):
            for dy in (-.61, .61):
                add(table, "geom", type="box", size=".035 .035 .365", pos=f"{dx} {dy} .365", rgba=".17 .19 .22 1")
        for dy in (-.7525, .7525):
            add(table, "geom", type="box", size="1.37 .01 .001", pos=f"0 {dy} .761", rgba="1 1 1 1", contype="0", conaffinity="0")
        for dx in (-1.36, 1.36):
            add(table, "geom", type="box", size=".01 .7625 .001", pos=f"{dx} 0 .761", rgba="1 1 1 1", contype="0", conaffinity="0")
        add(table, "geom", type="box", size="1.37 .0015 .001", pos="0 0 .761", rgba="1 1 1 1", contype="0", conaffinity="0")
        add(table, "geom", name="tennis_net", type="box", size=".006 .915 .07625", pos="0 0 .83625", rgba=".7 .75 .8 .65")
        add(table, "geom", type="box", size=".007 .915 .004", pos="0 0 .9125", rgba="1 1 1 1", contype="0", conaffinity="0")
        add_feeder(root, world)
        racket = add(world, "body", name="tennis_racket", pos=".57 -.22 .770")
        add(racket, "freejoint", name="tennis_racket_free")
        add(racket, "geom", name="tennis_racket_face", type="cylinder", size=".085 .006", mass=".14", rgba=".8 .07 .055 1", friction="1 .005 .0001")
        add(racket, "geom", name="tennis_racket_handle", type="box", size=".055 .014 .009", pos="-.13 0 0", mass=".035", rgba=".56 .32 .13 1", friction="1 .005 .0001")
        ball = add(world, "body", name="tennis_ball", pos="3.05 0 -1")
        add(ball, "freejoint", name="tennis_ball_free")
        add(ball, "geom", name="tennis_ball_geom", type="sphere", size=".02", mass=".0027", rgba="1 .8 .08 1", contype="0", conaffinity="0")
        # Native TennisTask owns ball flight/contact; never double-integrate it.
        size = root.find("size")
        if size is None:
            size = add(root, "size")
        size.set("nuserdata", "16")
        custom = root.find("custom")
        if custom is None:
            custom = add(root, "custom")
        # delay, interval, launch speed, Cd, air density, table restitution
        add(custom, "numeric", name="tennis_config", data="7 5 4.6 .5 1.204 .9")
        equality = root.find("equality")
        if equality is None:
            equality = add(root, "equality")
        # Three-finger G1 adaptation of a shakehand grip: index along the blade,
        # thumb on the opposite face, middle finger around the handle.
        add(custom, "numeric", name="tennis_grip_fingers",
            data="-.35 .85 .85 -.2 -.05 -.75 -1.6")
        # Upstream finger position actuators have no damping. The deliberate
        # holding pose otherwise excites finger oscillation after acquisition.
        for actuator in root.findall("actuator/position"):
            if "hand_" in actuator.get("name", ""):
                actuator.set("kp", "5")
        for joint in root.iter("joint"):
            if "hand_" in joint.get("name", ""):
                # Joint damping is integrated implicitly with the 1 ms Euler
                # step; actuator velocity feedback is explicit and unstable
                # for the distal fingers' very small inertia at this rate.
                joint.set("damping", ".03")
        tilt = math.radians(25)
        q0 = math.cos(math.pi/4-tilt/2)/math.sqrt(2)
        q1 = math.sin(math.pi/4-tilt/2)/math.sqrt(2)
        for side, sign in (("left", 1), ("right", -1)):
            # Handle crosses the palm toward the index-finger side (+local Z),
            # with a small forward tilt. Blade normal is the palm's local Y.
            position = (.13+.13*math.sin(tilt), -sign*.032, -.02+.13*math.cos(tilt))
            quaternion = (q0, -sign*q0, -q1, -sign*q1)
            add(equality, "weld", name=f"tennis_grip_{side}",
                body1=f"{side}_wrist_yaw_link", body2="tennis_racket", active="false",
                relpose=" ".join(str(x) for x in (*position, *quaternion)),
                solref=".002 1", solimp=".99 .999 .001")
        contact = root.find("contact")
        if contact is None:
            contact = add(root, "contact")
        for body in root.iter("body"):
            if any(g.get("contype") == "4" for g in body.findall("geom")):
                # Avoid fighting the holding constraint with finger/palm penetration.
                # Exact touching is still queried explicitly by native TennisTask.
                add(contact, "exclude", body1=body.get("name"), body2="tennis_racket")
    if scene in ("cup", "rubik"):
        add_workcell(root)
    if scene == "rubik":
        add_puzzle(root)
    add_collection_scene(root, scene)
    if hands and geometry != "source":
        configure_props(world)
    xml = ET.tostring(root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml), xml

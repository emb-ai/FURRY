"""Build scenes around the unmodified upstream G1 description."""
import math
import xml.etree.ElementTree as ET
import mujoco
from .controller import UPSTREAM


def add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def make_model(scene="lab", hands=True):
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
    root.find("compiler").set("meshdir", str(source.parent / "meshes"))
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
    add(torso, "camera", name="ego", pos=".08 0 .43", xyaxes="0 -1 0 .5 0 .8660254", fovy="80")
    if scene in ("cup", "lab"):
        x, y = (.65, 0.) if scene == "cup" else (.8, -.8)
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
    xml = ET.tostring(root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml), xml

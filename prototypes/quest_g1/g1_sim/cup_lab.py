"""Authored cup workcell, in metres, shared by MuJoCo and Quest GLES.

The room is presentation geometry, not an invisible obstacle course. The bench
and hollow mug have physical surfaces; all decorative meshes have zero mass.
"""
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

STYLE = "cup-lab-v2"
GOAL = (.43, 0., .75, .055)
CUP_XY_SCALE = .75
CUP_DIAMETER_M = .086 * CUP_XY_SCALE
CUP_MASS_KG = .240 * CUP_XY_SCALE**2

# Compact geometric lettering survives the native untextured mesh stream.
GLYPHS = {
    "F": ["1111", "1000", "1000", "1110", "1000", "1000", "1000"],
    "U": ["1001", "1001", "1001", "1001", "1001", "1001", "0110"],
    "R": ["1110", "1001", "1001", "1110", "1010", "1001", "1001"],
    "Y": ["1001", "1001", "1001", "0110", "0010", "0010", "0010"],
    "L": ["1000", "1000", "1000", "1000", "1000", "1000", "1111"],
    "A": ["0110", "1001", "1001", "1111", "1001", "1001", "1001"],
    "B": ["1110", "1001", "1001", "1110", "1001", "1001", "1110"],
    "0": ["0110", "1001", "1011", "1001", "1101", "1001", "0110"],
    "1": ["0010", "0110", "0010", "0010", "0010", "0010", "0111"],
    "P": ["1110", "1001", "1001", "1110", "1000", "1000", "1000"],
    "C": ["0111", "1000", "1000", "1000", "1000", "1000", "0111"],
    "E": ["1111", "1000", "1000", "1110", "1000", "1000", "1111"],
    " ": ["0000"]*7,
}


def el(parent, tag, **attrs):
    def text(v):
        return " ".join(format(x, ".9g") for x in v) if isinstance(v, (tuple, list)) else str(v)
    return ET.SubElement(parent, tag, {k: text(v) for k, v in attrs.items()})


def mesh(asset, name, points, faces):
    el(asset, "mesh", name=name, vertex=[x for p in points for x in p],
       face=[x for f in faces for x in f], smoothnormal="true")
    return name


def lathe(asset, name, profile, slices=64):
    points = [(r*math.cos(a*2*math.pi/slices), r*math.sin(a*2*math.pi/slices), z)
              for r, z in profile for a in range(slices)]
    faces = []
    for j in range(len(profile)-1):
        for i in range(slices):
            a, b = j*slices+i, j*slices+(i+1) % slices
            c, d = b+slices, a+slices
            for f in [(a, b, c), (a, c, d)]:
                if profile[j][0] != 0 or profile[j+1][0] != 0:
                    faces.append(f)
    return mesh(asset, name, points, faces)


def handle(asset, name, steps=32, sides=12):
    # A generous open loop: 40 mm clear horizontal reach, 48 mm clear height.
    points = []
    for i in range(steps+1):
        a = math.pi*i/steps
        center = (.039+.052*math.sin(a), 0, .050+.030*math.cos(a))
        tangent = (.052*math.cos(a), 0, -.030*math.sin(a))
        length = math.hypot(tangent[0], tangent[2])
        n = (-tangent[2]/length, 0, tangent[0]/length)
        for j in range(sides):
            b = 2*math.pi*j/sides
            points.append((center[0]+.006*math.cos(b)*n[0],
                           .006*math.sin(b), center[2]+.006*math.cos(b)*n[2]))
    faces = []
    for i in range(steps):
        for j in range(sides):
            a, b = i*sides+j, i*sides+(j+1) % sides
            faces.extend([(a, b, b+sides), (a, b+sides, a+sides)])
    # End caps are inside the cup wall.
    for j in range(1, sides-1):
        faces.extend([(0, j+1, j), (steps*sides, steps*sides+j, steps*sides+j+1)])
    return mesh(asset, name, points, faces)


def rounded_slab(asset, name):
    points = []
    rings = [(.314, .444, .019, -.025), (.320, .450, .025, -.019),
             (.320, .450, .025, .019), (.314, .444, .019, .025)]
    for hx, hy, r, z in rings:
        for sx, sy, start in [(1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)]:
            for j in range(9):
                a = math.radians(start+j*90/8)
                points.append((sx*(hx-r)+r*math.cos(a), sy*(hy-r)+r*math.sin(a), z))
    n = 36
    faces = []
    for k in range(3):
        for j in range(n):
            a, b = k*n+j, k*n+(j+1) % n
            faces.extend([(a, b, b+n), (a, b+n, a+n)])
    for j in range(1, n-1):
        faces.extend([(0, j+1, j), (3*n, 3*n+j, 3*n+j+1)])
    return mesh(asset, name, points, faces)


def lettering(asset, name, text, pixel):
    points, faces = [], []
    for c, char in enumerate(text):
        for row, line in enumerate(GLYPHS[char]):
            for run in re.finditer("1+",line):
                col,width=run.start(),len(run.group())
                x, y, n = (c*5+col)*pixel, (6-row)*pixel, len(points)
                # Thin closed boxes; map this XY text plane to the sign in MJCF.
                for z in (0, pixel*.12):
                    points.extend([(x, y, z), (x+pixel*(width-.12), y, z),
                                   (x+pixel*(width-.12), y+pixel*.88, z), (x, y+pixel*.88, z)])
                for quad in [(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]:
                    a,b,c1,d = [n+i for i in quad]
                    faces.extend([(a,b,c1),(a,c1,d)])
    return mesh(asset, name, points, faces)


def decoration(parent, name, kind="box", material="lab_plaster", **attrs):
    attrs.setdefault("group", 2)
    return el(parent, "geom", name=name, type=kind, material=material,
              contype=0, conaffinity=0, density=0, **attrs)


def add_workcell(root):
    asset, world = root.find("asset"), root.findall("worldbody")[-1]
    # Leave space for the workcell inside the existing 100k per-eye budget.
    # Only zero-density exterior meshes change; dynamics and colliders do not.
    for name in ("torso_link", "head_link"):
        exterior=asset.find(f"mesh[@name='{name}']")
        if exterior is None:
            exterior=next((m for m in asset.findall("mesh") if Path(m.get("file", "")).stem==name),None)
        if exterior is not None and exterior.get("file", "").endswith(".obj"):
            exterior.set("name",name)
            exterior.set("file",str(Path(__file__).resolve().parents[1]/"assets/g1/visual/cup_lab"/("cup-lab-"+name+".obj")))
    palette = {
        "lab_plaster": ((.78, .79, .75, 1), .05, .1),
        "lab_floor": ((.38, .43, .43, 1), .1, .15),
        "lab_metal": ((.12, .17, .18, 1), .35, .4),
        "lab_wood": ((.61, .46, .31, 1), .18, .2),
        "lab_ceramic": ((.055, .46, .47, 1), .45, .65),
        "lab_ceramic_inside": ((.70, .85, .79, 1), .4, .65),
        "lab_accent": ((.13, .39, .39, 1), .15, .25),
        "lab_goal": ((.20, .58, .51, 1), .1, .1),
        "lab_window": ((.64, .78, .82, 1), .1, .2),
        "lab_light": ((.98, .91, .72, 1), 0, 0),
        "lab_white": ((.92, .93, .85, 1), .15, .2),
    }
    for name, (rgba, specular, shininess) in palette.items():
        el(asset, "material", name=name, rgba=rgba, specular=specular, shininess=shininess,
           emission=.6 if name == "lab_light" else 0)
    floor = root.find(".//geom[@name='floor']")
    floor.set("material", "lab_floor")
    floor.set("rgba", ".38 .43 .43 1")
    for light in world.findall("light"):
        world.remove(light)
    el(world, "light", pos=(-1, -2, 3.5), dir=(.25, .4, -1),
       diffuse=(.75, .72, .65), ambient=(.25, .28, .29), specular=(.3, .3, .3), directional="true")
    el(world, "light", pos=(2.8, 1, 2.8), dir=(-1, -.2, -.6),
       diffuse=(.35, .43, .48), castshadow="false")
    visual = root.find("visual")
    quality = visual.find("quality")
    if quality is None:
        quality = el(visual, "quality")
    quality.set("shadowsize", "2048")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = el(visual, "headlight")
    headlight.set("active", "0")
    # Open operator side; distant decorative walls leave the work area clear.
    room = el(world, "body", name="cup_lab_room")
    decoration(room, "lab_back_wall", pos=(3.25, 0, 1.4), size=(.04, 3, 1.4))
    for y in (-3, 3):
        label = "left" if y > 0 else "right"
        decoration(room, "lab_"+label+"_wall", pos=(.5, y, 1.4), size=(2.75, .04, 1.4))
        decoration(room, "lab_"+label+"_skirting", pos=(.5, y*.986, .08), size=(2.75, .016, .08), material="lab_metal")
        decoration(room, "lab_"+label+"_stripe", pos=(.5, y*.984, 1.04), size=(2.75, .005, .028), material="lab_accent")
    decoration(room, "lab_back_skirting", pos=(3.20, 0, .08), size=(.012, 3, .08), material="lab_metal")
    decoration(room, "lab_sign_board", pos=(3.18, -1.92, 1.74), size=(.02, .57, .22), material="lab_accent")
    sign = lettering(asset, "lab_furry_sign", "FURRY LAB", .022)
    decoration(room, "lab_sign_letters", kind="mesh", mesh=sign, pos=(3.15, -1.45, 1.69),
               quat=(.5, .5, -.5, -.5), material="lab_white")
    # Window panels, mullions, and a warm overhead strip establish depth.
    for j in range(4):
        y = -.75+j*.5
        decoration(room, f"lab_window_{j}", pos=(3.195, y, 1.84), size=(.012, .235, .50), material="lab_window")
    for y in (-1.01, -.50, 0, .50, 1.01):
        decoration(room, f"lab_mullion_{y}", pos=(3.17, y, 1.84), size=(.018, .013, .53), material="lab_white")
    for z in (1.31, 2.37):
        decoration(room, f"lab_window_frame_{z}", pos=(3.17, 0, z), size=(.018, 1.03, .018), material="lab_white")
    for y in (-.9, .9):
        decoration(room, f"lab_ceiling_rail_{y}", pos=(.9, y, 2.73), size=(1.0, .065, .025), material="lab_metal")
        decoration(room, f"lab_ceiling_light_{y}", pos=(.9, y, 2.70), size=(.96, .044, .008), material="lab_light")
    # Background storage: restrained detail, well outside the reachable zone.
    storage = el(world, "body", name="cup_lab_storage", pos=(2.70, 1.75, 0))
    decoration(storage, "lab_cabinet", pos=(0, 0, .43), size=(.29, .72, .39), material="lab_white")
    decoration(storage, "lab_cabinet_top", pos=(0, 0, .84), size=(.32, .75, .022), material="lab_wood")
    for j, y in enumerate((-.48, 0, .48)):
        decoration(storage, f"lab_door_{j}", pos=(-.299, y, .44), size=(.008, .228, .35))
        decoration(storage, f"lab_pull_{j}", pos=(-.32, y-.14, .62), size=(.014, .045, .008), material="lab_metal")
    decoration(storage, "lab_shelf", pos=(.12, 0, 1.46), size=(.20, .76, .022), material="lab_wood")
    for j, y in enumerate((-.48, 0, .45)):
        decoration(storage, f"lab_storage_bin_{j}", pos=(.12, y, 1.57), size=(.13, .16, .085),
                   material="lab_accent" if j == 1 else "lab_white")
    # Floor expansion joints are actual visible strips, including on Quest.
    for axis in (0, 1):
        for j in range(-2, 4):
            p = [0, 0, .0003]; p[axis] = j
            s = [3.2, 3, .00015]; s[axis] = .003
            decoration(room, f"lab_floor_joint_{axis}_{j}", pos=p, size=s, material="lab_metal")

    table, cup = root.find(".//body[@name='table']"), root.find(".//body[@name='cup']")
    for geom in list(table.findall("geom")):
        table.remove(geom)
    slab = rounded_slab(asset, "lab_bench_top")
    el(table, "geom", name="tabletop", type="mesh", mesh=slab, pos=(0, 0, .725),
       material="lab_wood", friction=".8 .005 .0001")
    for j, (dx, dy) in enumerate(((-.26, -.39), (-.26, .39), (.26, -.39), (.26, .39))):
        el(table, "geom", name=f"lab_bench_leg_{j}", type="box", pos=(dx, dy, .35),
           size=(.024, .024, .35), material="lab_metal")
        decoration(table, f"lab_bench_foot_{j}", pos=(dx, dy, .016), size=(.031, .031, .015), material="lab_metal")
    for y in (-.39, .39):
        decoration(table, f"lab_bench_apron_{y}", pos=(0, y, .657), size=(.26, .016, .041), material="lab_metal")
    for x in (-.26, .26):
        decoration(table, f"lab_bench_cross_{x}", pos=(x, 0, .18), size=(.015, .39, .015), material="lab_metal")
    # Inlaid goal is flat, with no new collision step on the tabletop.
    ring = lathe(asset, "lab_goal_ring", [(.062, 0), (.065, .0004), (.062, .0008), (.059, .0004), (.062, 0)])
    decoration(table, "cup_place_ring", kind="mesh", mesh=ring, pos=(GOAL[0]-.65, GOAL[1], .7503), material="lab_goal")
    for dy in (-.010, .010):
        decoration(table, f"cup_place_tick_{dy}", pos=(GOAL[0]-.65, GOAL[1]+dy, .7506), size=(.018, .0012, .00015), material="lab_goal")
    label = lettering(asset, "lab_place_label", "PLACE", .004)
    decoration(table, "cup_place_label", kind="mesh", mesh=label, pos=(GOAL[0]-.65-.08, GOAL[1]-.045, .7505),
               euler=(0,0,-math.pi/2), material="lab_goal")

    for geom in list(cup.findall("geom")):
        cup.remove(geom)
    # Collision wall and bottom carry all .240 kg; visual shells carry zero.
    el(cup, "geom", name="cup_bottom", type="cylinder", size=(.043, .004), pos=(0, 0, .004),
       mass=.06, friction="1 .005 .0001", group=3, rgba=".2 .6 .4 .35")
    for j in range(32):
        a = 2*math.pi*j/32
        el(cup, "geom", name=f"cup_wall_{j}", type="box", size=(.0045, .0040, .044),
           pos=(.0395*math.cos(a), .0395*math.sin(a), .050), euler=(0, 0, a),
           mass=.144/32, friction="1 .005 .0001", group=3, rgba=".2 .6 .4 .35")
    for j in range(12):
        def center(a):
            return (.039+.052*math.sin(a), 0, .050+.030*math.cos(a))
        a, b = center(math.pi*j/12), center(math.pi*(j+1)/12)
        el(cup, "geom", name=f"cup_handle_{j}", type="capsule", size=.006, fromto=(*a, *b),
           mass=.036/12, friction="1 .005 .0001", group=3, rgba=".2 .6 .4 .35")
    shell = lathe(asset, "lab_mug_shell", [(0, .001), (.035, .001), (.040, .003),
        (.042, .008), (.043, .085), (.043, .091), (.042, .095), (.039, .096),
        (.036, .095), (.0355, .091), (.0355, .014), (.032, .010), (0, .010)])
    decoration(cup, "cup_ceramic_shell", kind="mesh", mesh=shell, material="lab_ceramic", group=1)
    grip = handle(asset, "lab_mug_handle")
    decoration(cup, "cup_ceramic_handle", kind="mesh", mesh=grip, material="lab_ceramic", group=1)
    # Compact mug: 64.5 mm body diameter, 96 mm tall, 135 g. Scale both
    # rendered surfaces and collisions; preserve the open cavity/handle.
    # These are FURRY dimensions, not claimed TWIST2 prop measurements.
    for geom in cup.findall("geom"):
        for field in ("pos", "fromto"):
            if geom.get(field):
                values=list(map(float,geom.get(field).split()))
                values=[v*CUP_XY_SCALE if i%3<2 else v for i,v in enumerate(values)]
                geom.set(field," ".join(map(str,values)))
        if geom.get("size"):
            values=list(map(float,geom.get("size").split()))
            axes=2 if geom.get("type")=="box" else 1
            values=[v*CUP_XY_SCALE if i<axes else v for i,v in enumerate(values)]
            geom.set("size"," ".join(map(str,values)))
        if geom.get("mass"):geom.set("mass",str(float(geom.get("mass"))*CUP_XY_SCALE**2))
        if geom.get("type")=="mesh":
            asset.find("mesh[@name='%s']"%geom.get("mesh")).set("scale",f"{CUP_XY_SCALE} {CUP_XY_SCALE} 1")
        elif geom.get("contype")!="0":
            # Rigid ceramic contact; the old soft contact allowed continuing
            # closure to roll the cup out. Same formulation as the puzzle.
            geom.set("solref",".004 1");geom.set("solimp",".98 .995 .001");geom.set("condim","4")
    custom = root.find("custom")
    if custom is None:
        custom = el(root, "custom")
    el(custom, "numeric", name="cup_lab_goal", data=GOAL)
    el(custom, "text", name="cup_lab_style", data=STYLE)

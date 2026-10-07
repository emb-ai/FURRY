# G1 Hybrid Geometry

The default desktop and Android model retains the same articulated three-finger
hands: 29 body actuators and 14 finger actuators. Mesh comparisons use that same
skeleton and pose, never the fixed-hand MJX visual model.

## Visuals

The 49 versioned OBJ assets contain **88,047 triangles**, versus 629,338 in the
source and 66,509 in the previous Quest export. The complete Android scene is
**89,629 triangles per eye**, including props and UI, below the 100,000 limit.

Complex torso, pelvis, knee, palm and finger parts are rebuilt as voxel-remeshed
shells, smoothed and reduced with per-part budgets. Simpler parts use welded,
conservative topology reduction. This is reproducible automated remodelling,
not hand-authored CAD reconstruction. Small internal details can disappear.
Explicit corner normals preserve smooth surfaces and creases over 40 degrees;
Android exports those normals instead of flattening every triangle. Mesh frames,
joint axes, body inertias and control parameters are unchanged.

`visual/manifest.json` records source/output SHA-256, recipe, budgets and Blender
version. Its per-part `source_triangles` counts are measured after Blender's STL
import cleanup, not raw STL counts. Four full-body comparison views currently have silhouette IoU above
0.998; this is a projection check, not a bound on every surface detail.

## Collisions

- **25 body primitives** from Menagerie `g1_mjx.xml`: 21 capsules, two spheres,
  two boxes. The two whole-hand capsules are deliberately omitted.
- **16 independent convex hand meshes**: one palm and seven finger links per
  hand, 2,444 triangles total, 38-224 triangles each. No hull spans a joint.
- Finger envelopes stay within 0.75 mm of the original link's convex hull;
  palms within 1.5 mm. Supporting planes include 0.1 mm outward padding and
  enclose source vertices. These are convex-envelope errors, not errors relative
  to concavities in the visual CAD surface.
- Body self-collision uses 19 explicit MJX-derived pairs. Hands collide with
  props, floor, body and other hand links subject to MuJoCo's parent filtering.
  Props also collide with body primitives. Sole capsules, not the overlapping
  foot boxes, contact the floor; foot boxes remain available for object contact.
- Categories are floor=1, body=2, hand=4, foot-box=8, prop=16. Collision geometry
  is group 3 and excluded from Android visual streams.

Only the collision shapes and contact selection come from MJX, not its solver,
timestep, articulation or actuation. This remains the existing native MuJoCo
runtime, not a migration to the MJX backend. A new prop must use the prop category
or be registered in `configure_props` to preserve this contact policy.

## Provenance

The comparison fetched these upstream revisions on 2026-10-07:

- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/f054586a8e90465d49ee5be15335c4a0c7f57caf/unitree_g1):
  `f054586a8e90465d49ee5be15335c4a0c7f57caf`.
- [Unitree ROS](https://github.com/unitreerobotics/unitree_ros/tree/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/g1_description):
  `5994d4faef0a9cadd3287f8de0199a67eeb2a259`.
- Runtime skeleton and source meshes: TWIST2
  `b06178f19a22f2138cbd31f60c6d494bc263f67d`.

The 49 matching source meshes are byte-identical across these checkouts.
Unitree's BSD-3-Clause terms are preserved in [LICENSE](LICENSE), also packaged
with Android assets. Vendor repositories are not modified.

## Authoring And Verification

Normal runtime and Android builds read committed assets and need no Blender,
SciPy, trimesh or simplification package. Optional authoring commands, run from
`prototypes/quest_g1` with the pinned vendor checkout present:

```sh
.venv/bin/python -m pip install -r requirements-geometry.txt
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --python scripts/remodel_visuals.py
.venv/bin/python scripts/build_collision_assets.py --menagerie /path/to/mujoco_menagerie
.venv/bin/python scripts/inspect_geometry.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/prepare_android.py
```

Visual authoring was performed with Blender 5.0.1. Inspection writes ignored
`outputs/geometry/{comparison,hand-open,hand-closed}.png` and `report.json`, using
the original articulated model, reconstructed previous Quest visuals, new
visuals and new collisions in matching poses. The report recomputes hand fitting
errors independently of the asset metadata. Optional inspection needs OpenGL.

Tests cover unchanged articulation/dynamics, closed convex proxies enclosing
source vertices, contact on every palm/finger link, sole-floor filtering, visual
budgets and corner normals, plus the existing 60-second balance and prop tests.
Native tracking and recording checks use the regenerated Android scene.

For controlled comparisons, Python `make_model` accepts `geometry="source"` or
`geometry="collision_only"`; the default is `"hybrid"`. `--fixed-hands` retains
the previous fixed-hand model and is not the comparison baseline. Headset FPS,
thermal performance and a successful dynamic grasp are not established by these
offline checks and must be validated separately.

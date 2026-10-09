"""Package independent Quest scenes with matching visuals and episode metadata."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import mujoco

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from g1_sim.scene import make_model
from g1_sim.visuals import export_visual_meshes

SCENES = ("stand", "cup", "push_t")
HUD_MAX_TRIANGLES = 8192
OPERATOR_SKELETON_MAX_TRIANGLES = 2048  # 1024 unique triangles, two depth passes
MAX_TRIANGLES_PER_EYE = 100000
MAX_SCENE_GEOMETRIES = 2048
# Mirrors quest_runtime.cpp Geometry: 16 slices, eight sphere rings, and
# two cylinder/capsule caps. Unsupported MuJoCo arrow/line types are skipped.
PRIMITIVE_TRIANGLES = {
    mujoco.mjtGeom.mjGEOM_BOX: 12, mujoco.mjtGeom.mjGEOM_PLANE: 12,
    mujoco.mjtGeom.mjGEOM_SPHERE: 256, mujoco.mjtGeom.mjGEOM_ELLIPSOID: 256,
    mujoco.mjtGeom.mjGEOM_CYLINDER: 64, mujoco.mjtGeom.mjGEOM_CAPSULE: 64,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _copy_meshes(root: ET.Element, assets: Path, known_meshes: dict) -> list[Path]:
    compiler = root.find("compiler")
    source_directory = Path(compiler.get("meshdir", "."))
    destination_directory = assets / "meshes"
    destination_directory.mkdir(exist_ok=True)
    paths = []
    for mesh in root.iter("mesh"):
        mesh_file = mesh.get("file")
        if not mesh_file:
            continue
        source = (source_directory / mesh_file).resolve()
        destination = destination_directory / source.name
        digest = _sha256(source)
        previous = known_meshes.get(source.name)
        if previous is not None and previous != digest:
            raise ValueError(f"Different scene meshes share a filename: {source.name}")
        if previous is None:
            # Use the same basenames as the legacy lab assets. Do not remove
            # files another packaged scene still references.
            if not destination.exists() or _sha256(destination) != digest:
                shutil.copy2(source, destination)
            known_meshes[source.name] = digest
        mesh.set("file", source.name)
        paths.append(destination)
    compiler.set("meshdir", "meshes")
    return paths


def _export_visuals(model: mujoco.MjModel, path: Path, scene_name: str) -> dict:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scene = mujoco.MjvScene(model, maxgeom=MAX_SCENE_GEOMETRIES)
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    mujoco.mjv_updateScene(model, data, option, None, mujoco.MjvCamera(),
                         mujoco.mjtCatBit.mjCAT_ALL, scene)
    # MuJoCo's renderer dataid stores two entries per mesh. The native GLES
    # renderer uses this same index to address the exported model-order stream.
    mesh_usage = Counter(g.dataid // 2 for g in scene.geoms[:scene.ngeom]
                         if g.type == mujoco.mjtGeom.mjGEOM_MESH)
    primitive_triangles = 2 + sum(
        PRIMITIVE_TRIANGLES.get(g.type, 0) for g in scene.geoms[:scene.ngeom]
        if g.type != mujoco.mjtGeom.mjGEOM_PLANE)
    counts = export_visual_meshes(model, path, mesh_usage)
    scene_triangles = primitive_triangles + sum(counts[i] * uses for i, uses in mesh_usage.items())
    # Passthrough can be disabled from the menu, exposing the physical floor.
    floor_triangles = PRIMITIVE_TRIANGLES[mujoco.mjtGeom.mjGEOM_PLANE]
    interface_triangles = HUD_MAX_TRIANGLES + OPERATOR_SKELETON_MAX_TRIANGLES
    total = scene_triangles + floor_triangles + interface_triangles
    if total >= MAX_TRIANGLES_PER_EYE:
        raise ValueError(f"Quest scene {scene_name} exceeds visual budget: {total}")
    # Collision debug replaces the robot's exterior shells with colliders;
    # props retain group 0 and stay visible. This mirrors PublishGeometry.
    option.geomgroup[1] = 0
    option.geomgroup[3] = 1
    mujoco.mjv_updateScene(model, data, option, None, mujoco.MjvCamera(),
                         mujoco.mjtCatBit.mjCAT_ALL, scene)
    debug_scene_triangles = 2
    collision_triangles = 0
    for geom in scene.geoms[:scene.ngeom]:
        if geom.rgba[3] < .01 or geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        collision = (geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid >= 0
                     and model.geom_group[geom.objid] == 3)
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            index = geom.dataid // 2
            triangles = int(model.mesh_facenum[index]) if collision else counts[index]
        else:
            triangles = PRIMITIVE_TRIANGLES.get(geom.type, 0)
        debug_scene_triangles += triangles
        if collision:
            collision_triangles += triangles
    debug_collision_total = debug_scene_triangles + floor_triangles + interface_triangles
    # Contact points are cylinders in mjvScene. This is a conservative slot
    # bound, not a measured maximum ncon: force arrows also consume slots but
    # are not drawn by the current native primitive renderer. Export the bound
    # separately from the runtime's enforced optional-geometry triangle cap.
    contact_capacity = max(0, MAX_SCENE_GEOMETRIES - scene.ngeom)
    contact_triangles = PRIMITIVE_TRIANGLES[mujoco.mjtGeom.mjGEOM_CYLINDER]
    debug_unbounded = debug_collision_total + contact_capacity * contact_triangles
    contact_budget = max(0, MAX_TRIANGLES_PER_EYE - debug_collision_total)
    collision = json.loads((ROOT / "assets/g1/collision.json").read_text(encoding="utf-8"))
    return {
        "scene": scene_name,
        "maximum_triangles_per_eye": MAX_TRIANGLES_PER_EYE,
        "scene_triangles_per_eye": scene_triangles,
        "hud_max_triangles": HUD_MAX_TRIANGLES,
        "operator_skeleton_max_triangles": OPERATOR_SKELETON_MAX_TRIANGLES,
        "floor_max_triangles": floor_triangles,
        "total_max_triangles_per_eye": total,
        "debug_scene_triangles_per_eye": debug_scene_triangles,
        "debug_collision_triangles_per_eye": collision_triangles,
        "debug_collision_total_max_triangles_per_eye": debug_collision_total,
        "debug_contact_point_triangles": contact_triangles,
        "debug_contact_point_capacity_bound": contact_capacity,
        "debug_contact_triangle_budget": contact_budget,
        "debug_unbounded_total_upper_bound_triangles_per_eye": debug_unbounded,
        "debug_total_max_triangles_per_eye": min(debug_unbounded, MAX_TRIANGLES_PER_EYE),
        "debug_contacts_require_render_cap": debug_unbounded > MAX_TRIANGLES_PER_EYE,
        "debug_robot_visual_shells": False,
        "primitive_triangles": primitive_triangles,
        "unique_visual_mesh_triangles": sum(counts),
        "source_visual_mesh_triangles": 629338,
        "method": "authored exterior shells with crease-aware normals",
        "mesh_instances": sum(mesh_usage.values()),
        "collision_model": "mjx_body_articulated_hands_v1",
        "body_primitives": len(collision["body"]),
        "hand_convex_meshes": len(collision["hands"]),
        "hand_collision_triangles": sum(len(h["faces"]) for h in collision["hands"]),
    }


def export_scenes(assets: Path) -> dict:
    """Export stand/cup/push_t after shared policy/GMR/metadata preparation.

    The legacy lab files remain available for desktop checks and old episodes.
    Each scene's stream and dimensions come from its separately loaded model.
    No platform font or graphics context is needed, including on Linux.
    """
    assets = Path(assets).resolve()
    shared_metadata = json.loads((assets / "recording_metadata.json").read_text(encoding="utf-8"))
    manifest = {"schema_version": 1, "default_scene": "stand", "scenes": {}}
    known_meshes = {}
    for scene_name in SCENES:
        _, xml = make_model(scene_name, hands=True)
        root = ET.fromstring(xml)
        pelvis = root.find(".//body[@name='pelvis']")
        for geom in pelvis.iter("geom"):
            if geom.get("contype") != "0":
                geom.set("group", "3")
        mesh_paths = _copy_meshes(root, assets, known_meshes)
        files = {
            "model": f"scene-{scene_name}.xml",
            "visual_meshes": f"visual_meshes-{scene_name}.bin",
            "recording_metadata": f"recording_metadata-{scene_name}.json",
            "mesh_budget": f"mesh_budget-{scene_name}.json",
        }
        model_path = assets / files["model"]
        ET.ElementTree(root).write(model_path, encoding="unicode")
        model = mujoco.MjModel.from_xml_path(str(model_path))
        budget = _export_visuals(model, assets / files["visual_meshes"], scene_name)
        _write_json(assets / files["mesh_budget"], budget)
        metadata = deepcopy(shared_metadata)
        metadata.update(scene=scene_name, scene_name=scene_name, model_file=files["model"],
                        visual_meshes_file=files["visual_meshes"], nq=model.nq, nv=model.nv,
                        nu=model.nu, joints=[model.joint(i).name for i in range(model.njnt)])
        # A legacy model hash with the wrong state dimensions would make an
        # episode ambiguous. Preserve policy, GMR and source-code provenance.
        metadata["sha256"] = {
            name: digest for name, digest in metadata.get("sha256", {}).items()
            if name not in ("scene.xml", "visual_meshes.bin") and not name.startswith("meshes/")
        }
        for path in [model_path, assets / files["visual_meshes"], *mesh_paths]:
            metadata["sha256"][path.relative_to(assets).as_posix()] = _sha256(path)
        _write_json(assets / files["recording_metadata"], metadata)
        manifest["scenes"][scene_name] = files | {"nq": model.nq, "nv": model.nv, "nu": model.nu}
    _write_json(assets / "scene_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    export_scenes(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "android/assets")

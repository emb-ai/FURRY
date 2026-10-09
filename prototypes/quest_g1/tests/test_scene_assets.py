"""Verify real scene isolation, replay provenance and per-model render streams."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from g1_sim.geometry import BODY, HAND, PROP
from g1_sim.visuals import mesh_stream
from scripts.scene_assets import OPERATOR_SKELETON_MAX_TRIANGLES, PRIMITIVE_TRIANGLES, export_scenes


class SceneAssetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.assets = Path(cls.temporary.name)
        (cls.assets / "policy.onnx").write_bytes(b"shared-policy")
        cls.shared = {
            "schema_version": 1, "retargeting": "native_gmr_meta_full_body",
            "physics_hz": 1000, "policy_hz": 100, "nq": 64, "nv": 61, "nu": 43,
            "joints": ["legacy-joint"], "body_tracking": {"lower_body": "estimated"},
            "sha256": {"scene.xml": "legacy", "visual_meshes.bin": "legacy",
                       "meshes/obsolete.obj": "legacy",
                       "policy.onnx": hashlib.sha256(b"shared-policy").hexdigest(),
                       "android/native/gmr.cpp": "shared-code-hash"},
        }
        (cls.assets / "recording_metadata.json").write_text(json.dumps(cls.shared))
        (cls.assets / "scene.xml").write_text("legacy model untouched")
        (cls.assets / "visual_meshes.bin").write_bytes(b"legacy visual stream untouched")
        cls.manifest = export_scenes(cls.assets)
        cls.models = {name: mujoco.MjModel.from_xml_path(str(cls.assets / files["model"]))
                      for name, files in cls.manifest["scenes"].items()}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_scene_objects_are_physically_separate(self):
        expected = {"stand": set(), "cup": {"table", "cup"},
                    "push_t": {"t_box", "t_target"}}
        all_props = {"table", "cup", "t_box", "t_target"}
        for name, model in self.models.items():
            with self.subTest(scene=name):
                props = {model.body(i).name for i in range(model.nbody)} & all_props
                self.assertEqual(props, expected[name])
                self.assertEqual(model.nu, 43)
                self.assertEqual(model.opt.timestep, .001)
                self.assertEqual(model.opt.solver, mujoco.mjtSolver.mjSOL_NEWTON)
                dynamic_prop = "cup" if name == "cup" else "t_box" if name == "push_t" else None
                self.assertEqual((model.nq, model.nv), (57, 55) if dynamic_prop else (50, 49))
                if dynamic_prop:
                    body = model.body(dynamic_prop)
                    self.assertGreater(body.mass[0], 0)
                    joint = model.joint(dynamic_prop + "_free")
                    self.assertEqual(joint.type[0], mujoco.mjtJoint.mjJNT_FREE)
                    geoms = np.flatnonzero(model.geom_bodyid == body.id)
                    self.assertTrue(np.all(model.geom_contype[geoms] == PROP))
                    self.assertTrue(np.all(model.geom_conaffinity[geoms] > 0))
                if name == "push_t":
                    goal_geoms = np.flatnonzero(model.geom_bodyid == model.body("t_target").id)
                    self.assertTrue(np.all(model.geom_contype[goal_geoms] == 0))
                    self.assertTrue(np.all(model.geom_conaffinity[goal_geoms] == 0))
                hidden = np.flatnonzero(np.isin(model.geom_contype, [BODY, HAND]))
                self.assertTrue(np.all(model.geom_group[hidden] == 3))

    def test_metadata_identifies_matching_model_and_shared_policy(self):
        for name, files in self.manifest["scenes"].items():
            with self.subTest(scene=name):
                model = self.models[name]
                metadata = json.loads((self.assets / files["recording_metadata"]).read_text())
                self.assertEqual(metadata["scene"], name)
                self.assertEqual(metadata["model_file"], files["model"])
                self.assertEqual(metadata["visual_meshes_file"], files["visual_meshes"])
                self.assertEqual([metadata[key] for key in ("nq", "nv", "nu")],
                                 [model.nq, model.nv, model.nu])
                self.assertEqual(metadata["joints"], [model.joint(i).name for i in range(model.njnt)])
                self.assertEqual(metadata["body_tracking"], self.shared["body_tracking"])
                hashes = metadata["sha256"]
                self.assertNotIn("scene.xml", hashes)
                self.assertNotIn("visual_meshes.bin", hashes)
                self.assertNotIn("meshes/obsolete.obj", hashes)
                self.assertEqual(hashes["android/native/gmr.cpp"], "shared-code-hash")
                self.assertEqual(hashes["policy.onnx"], self.shared["sha256"]["policy.onnx"])
                paths = [files["model"], files["visual_meshes"]]
                xml = ET.parse(self.assets / files["model"]).getroot()
                self.assertEqual(xml.find("compiler").get("meshdir"), "meshes")
                paths += ["meshes/" + mesh.get("file") for mesh in xml.iter("mesh") if mesh.get("file")]
                for path in paths:
                    self.assertEqual(hashes[path], hashlib.sha256((self.assets / path).read_bytes()).hexdigest())

    def test_visual_stream_matches_model_order_and_triangle_budget(self):
        for name, files in self.manifest["scenes"].items():
            with self.subTest(scene=name):
                model = self.models[name]
                budget = json.loads((self.assets / files["mesh_budget"]).read_text())
                self.assertEqual(budget["scene"], name)
                self.assertEqual(budget["total_max_triangles_per_eye"],
                                 budget["scene_triangles_per_eye"] + budget["hud_max_triangles"]
                                 + budget["operator_skeleton_max_triangles"] + budget["floor_max_triangles"])
                self.assertEqual(budget["operator_skeleton_max_triangles"], 2048)
                self.assertLess(budget["total_max_triangles_per_eye"], 100000)
                visible = {int(model.geom_dataid[i]) for i in range(model.ngeom)
                           if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
                           and model.geom_group[i] != 3}
                triangles = 0
                with (self.assets / files["visual_meshes"]).open("rb") as stream:
                    self.assertEqual(struct.unpack("<I", stream.read(4))[0], model.nmesh)
                    for mesh_id in range(model.nmesh):
                        vertex_count = struct.unpack("<I", stream.read(4))[0]
                        self.assertEqual(vertex_count % 3, 0)
                        packed = np.frombuffer(stream.read(vertex_count * 24), dtype="<f4")
                        if mesh_id in visible:
                            np.testing.assert_array_equal(packed.reshape(-1, 3, 6), mesh_stream(model, mesh_id))
                        else:
                            self.assertEqual(vertex_count, 0)
                        triangles += vertex_count // 3
                    self.assertEqual(stream.read(), b"")
                self.assertEqual(triangles, budget["unique_visual_mesh_triangles"])

    def test_debug_budget_counts_hidden_collisions_and_bounds_contact_slots(self):
        self.assertEqual(OPERATOR_SKELETON_MAX_TRIANGLES, 2048)
        self.assertEqual(PRIMITIVE_TRIANGLES[mujoco.mjtGeom.mjGEOM_SPHERE], 16 * 8 * 2)
        self.assertEqual(PRIMITIVE_TRIANGLES[mujoco.mjtGeom.mjGEOM_CYLINDER], 16 * 2 + 16 * 2)
        for name, files in self.manifest["scenes"].items():
            with self.subTest(scene=name):
                model = self.models[name]
                budget = json.loads((self.assets / files["mesh_budget"]).read_text())
                collision_triangles = 0
                for geom_id in range(model.ngeom):
                    if model.geom_group[geom_id] != 3 or model.geom_rgba[geom_id, 3] < .01:
                        continue
                    if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
                        collision_triangles += int(model.mesh_facenum[model.geom_dataid[geom_id]])
                    else:
                        collision_triangles += PRIMITIVE_TRIANGLES.get(model.geom_type[geom_id], 0)
                self.assertEqual(budget["debug_collision_triangles_per_eye"], collision_triangles)
                self.assertEqual(budget["debug_collision_total_max_triangles_per_eye"],
                                 budget["debug_scene_triangles_per_eye"] + budget["hud_max_triangles"]
                                 + budget["operator_skeleton_max_triangles"] + budget["floor_max_triangles"])
                self.assertFalse(budget["debug_robot_visual_shells"])
                self.assertLess(budget["debug_scene_triangles_per_eye"], budget["scene_triangles_per_eye"])
                self.assertLess(budget["debug_collision_total_max_triangles_per_eye"], 100000)
                self.assertGreater(budget["debug_contact_point_capacity_bound"], 0)
                self.assertEqual(budget["debug_unbounded_total_upper_bound_triangles_per_eye"],
                                 budget["debug_collision_total_max_triangles_per_eye"]
                                 + budget["debug_contact_point_capacity_bound"] * 64)
                # Dense contact decoration requires the runtime's optional
                # geometry cap; physical contacts and stats remain unchanged.
                self.assertEqual(budget["debug_contact_triangle_budget"],
                                 100000 - budget["debug_collision_total_max_triangles_per_eye"])
                self.assertEqual(budget["debug_contacts_require_render_cap"],
                                 budget["debug_unbounded_total_upper_bound_triangles_per_eye"] > 100000)
                self.assertEqual(budget["debug_total_max_triangles_per_eye"],
                                 min(100000, budget["debug_unbounded_total_upper_bound_triangles_per_eye"]))

    def test_export_preserves_legacy_assets_and_has_explicit_manifest(self):
        self.assertEqual((self.assets / "scene.xml").read_text(), "legacy model untouched")
        self.assertEqual((self.assets / "visual_meshes.bin").read_bytes(), b"legacy visual stream untouched")
        self.assertEqual(json.loads((self.assets / "recording_metadata.json").read_text()), self.shared)
        self.assertEqual(json.loads((self.assets / "scene_manifest.json").read_text()), self.manifest)
        self.assertEqual(self.manifest["default_scene"], "stand")
        self.assertEqual(set(self.manifest["scenes"]), {"stand", "cup", "push_t"})


if __name__ == "__main__":
    unittest.main()

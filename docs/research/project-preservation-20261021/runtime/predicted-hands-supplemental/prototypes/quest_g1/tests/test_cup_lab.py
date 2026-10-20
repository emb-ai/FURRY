"""Physical mug cavity/handle, supported placement and native cue checks."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from g1_sim.controller import Controller
from g1_sim.scene import make_model


class CupLabTests(unittest.TestCase):
    def test_cavity_and_handle_hole_are_physically_open(self):
        _,xml=make_model("cup");root=ET.fromstring(xml)
        body=ET.SubElement(root.find("worldbody"),"body",name="probe")
        ET.SubElement(body,"freejoint",name="probe_joint")
        ET.SubElement(body,"geom",name="probe",type="sphere",size=".003",mass=".001",contype="16",conaffinity="31")
        m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"));d=mujoco.MjData(m)
        mujoco.mj_forward(m,d)
        cup=m.body("cup").id;origin=d.xpos[cup].copy();q=int(m.joint("probe_joint").qposadr[0]);probe=m.geom("probe").id
        for point,touch in [((0,0,.05),False),((.064,0,.05),False),((.0395,0,.05),True),((.091,0,.05),True)]:
            with self.subTest(point=point):
                d.qpos[q:q+3]=origin+point;d.qpos[q+3:q+7]=[1,0,0,0];mujoco.mj_forward(m,d)
                contacts=[c for c in d.contact if probe in (c.geom1,c.geom2) and cup in (m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2])]
                self.assertEqual(bool(contacts),touch)
        self.assertAlmostEqual(float(m.body_mass[cup]),.24,places=12)

    def test_new_mug_settles_on_bench_at_multiple_yaws(self):
        m,_=make_model("cup");d=mujoco.MjData(m);controller=Controller(m)
        q=int(m.joint("cup_free").qposadr[0])
        for yaw in (0,np.pi/2,np.pi,3*np.pi/2):
            with self.subTest(yaw=yaw):
                controller.initialize(d);d.qpos[0]=-2
                d.qpos[q:q+3]=[.53,0,.83];d.qpos[q+3:q+7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
                mujoco.mj_forward(m,d)
                for _ in range(1500):controller.step(d)
                self.assertTrue(np.isfinite(d.qpos).all())
                self.assertLess(abs(d.xpos[m.body("cup").id,2]-.75),.006)
                self.assertGreater(d.xmat[m.body("cup").id,8],.98)
                self.assertEqual(sum(w.number for w in d.warning),0)

    def test_room_decorations_do_not_add_physical_obstacles(self):
        m,_=make_model("cup")
        for body in ("cup_lab_room","cup_lab_storage"):
            geoms=np.flatnonzero(m.geom_bodyid==m.body(body).id)
            self.assertTrue(np.all(m.geom_contype[geoms]==0))
            self.assertTrue(np.all(m.geom_conaffinity[geoms]==0))
        self.assertEqual((m.nq,m.nv,m.nu),(57,55,43))

    def test_native_placement_cue(self):
        root=Path(__file__).resolve().parents[1];compiler=os.environ.get("CXX") or shutil.which("c++")
        self.assertIsNotNone(compiler)
        mj=Path(mujoco.__file__).parent
        library=next(mj.glob("libmujoco*.dylib" if sys.platform=="darwin" else "libmujoco.so*"))
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);_,xml=make_model("cup");(folder/"scene.xml").write_text(xml)
            binary=folder/"check_cup_lab"
            subprocess.run([compiler,"-std=c++17","-O2","-Wall","-Wextra","-Werror",
                "-I"+str(root/"android/native"),"-I"+str(mj/"include"),str(root/"tests/check_cup_lab.cpp"),str(library),"-o",str(binary)],check=True)
            env=dict(os.environ);env["DYLD_LIBRARY_PATH" if sys.platform=="darwin" else "LD_LIBRARY_PATH"]=str(mj)
            result=subprocess.run([str(binary),str(folder/"scene.xml")],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("Cup placement cue passed",result.stdout)


if __name__=="__main__":unittest.main()

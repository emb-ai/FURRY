import sys
from pathlib import Path
import os, subprocess, tempfile, unittest, json
import mujoco
from scripts.scene_assets import export_scenes

class CollectionTests(unittest.TestCase):
    def test_native_seeded_scenes_and_shots(self):
        root=Path(__file__).resolve().parents[1];mj=Path(mujoco.__file__).parent
        with tempfile.TemporaryDirectory() as tmp:
            assets=Path(tmp);(assets/'recording_metadata.json').write_text(json.dumps({'schema_version':1,'sha256':{}}));export_scenes(assets)
            binary=assets/'check_collection';lib=next(mj.glob('libmujoco*.dylib' if sys.platform == 'darwin' else 'libmujoco.so*'))
            subprocess.run(['c++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I'+str(root/'android/native'),'-I'+str(mj/'include'),str(root/'tests/check_collection.cpp'),str(lib),'-o',str(binary)],check=True)
            result=subprocess.run([str(binary),str(assets)],env=dict(os.environ, **{('DYLD_LIBRARY_PATH' if sys.platform == 'darwin' else 'LD_LIBRARY_PATH'): str(mj)}),capture_output=True,text=True)
            print(result.stdout,end='');self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_basket_opens_door_by_contact(self):
        from g1_sim.scene import make_model
        model,_=make_model('basket');data=mujoco.MjData(model)
        basket=model.joint('task_object_0_free');q=int(basket.qposadr[0]);v=int(basket.dofadr[0])
        hinge=model.joint('basket_door_hinge');hq=int(hinge.qposadr[0])
        data.qpos[q:q+3]=[1.36,0,.8];data.qvel[v]=1.5
        mujoco.mj_forward(model,data)
        contact=False
        for _ in range(300):
            mujoco.mj_step(model,data)
            for c in data.contact:
                bodies={int(model.geom_bodyid[g]) for g in c.geom if g>=0}
                if {model.body('task_object_0').id,model.body('basket_door').id}<=bodies:contact=True
        self.assertTrue(contact,'Basket passed through the door without contact')
        self.assertGreater(float(data.qpos[hq]),.02,'Contact did not open the passive door')
        self.assertLess(float(data.qpos[hq]),1.67)

    def test_door_swing_clears_destination_shelf(self):
        import numpy as np
        from g1_sim.scene import make_model
        m,_=make_model('basket');d=mujoco.MjData(m)
        q=int(m.joint('basket_door_hinge').qposadr[0])
        pair={m.body('basket_door').id,m.body('basket_destination').id}
        for angle in np.linspace(0,1.65,40):
            d.qpos[q]=angle;mujoco.mj_forward(m,d)
            for c in d.contact:
                bodies={int(m.geom_bodyid[g]) for g in c.geom if g>=0}
                self.assertFalse(bodies==pair and c.dist<0,'Door intersects destination shelf')

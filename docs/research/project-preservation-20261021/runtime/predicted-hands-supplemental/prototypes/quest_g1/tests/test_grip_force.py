"""Actual 1 ms finger dynamics and blocked motor effort in manipulation scenes."""
import unittest
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from g1_sim.scene import make_model


def fixture(xml):
    tree=ET.fromstring(xml)
    for parent in tree.iter():
        for child in list(parent):
            if child.tag=='freejoint' or (child.tag=='joint' and 'hand_' not in child.get('name','')):
                parent.remove(child)
    for actuator in list(tree.find('actuator')):
        if 'hand_' not in actuator.get('name',''):
            tree.find('actuator').remove(actuator)
    for tag in ('sensor','equality','contact','keyframe'):
        for element in tree.findall(tag):tree.remove(element)
    tree.find('option').set('gravity','0 0 0')
    for geom in tree.iter('geom'):
        geom.set('contype','0');geom.set('conaffinity','0')
    return mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'))


class GripForceTests(unittest.TestCase):
    def test_blocked_fingers_cannot_apply_full_fist_effort(self):
        for scene in ('cup','rubik'):
            with self.subTest(scene=scene):
                model,_=make_model(scene);data=mujoco.MjData(model)
                fingers=[i for i in range(model.nu) if 'hand_' in model.actuator(i).name]
                self.assertEqual(len(fingers),14)
                for aid in fingers:
                    joint=model.actuator_trnid[aid,0]
                    low,high=model.jnt_range[joint]
                    # Obstruct each finger well before its commanded angle.
                    data.qpos[model.jnt_qposadr[joint]]=low
                    data.ctrl[aid]=high
                mujoco.mj_forward(model,data)
                np.testing.assert_allclose(data.actuator_force[fingers],.35,atol=1e-12)

    def test_full_closure_settles_without_finger_ringing(self):
        for scene in ('cup','rubik'):
            with self.subTest(scene=scene):
                _,xml=make_model(scene);model=fixture(xml);data=mujoco.MjData(model)
                targets=[]
                for aid in range(model.nu):
                    name=model.actuator(aid).name;sign=1 if name.startswith('left') else -1
                    target=0 if 'thumb_0' in name else sign*(1 if '_1_' in name else 1.74) if 'thumb' in name else -sign*(1.57 if '_0_' in name else 1.74)
                    targets.append(target)
                velocities=[]
                for step in range(3000):
                    data.ctrl[:]=np.array(targets)*min(step*.002,1)
                    mujoco.mj_step(model,data)
                    if step>=2000:velocities.append(data.qvel.copy())
                self.assertLess(np.max(np.abs(velocities)),.01)
                self.assertEqual(sum(w.number for w in data.warning),0)
                np.testing.assert_allclose(data.qpos,np.array(targets),atol=.01)

    def test_existing_tennis_grip_and_stand_are_unchanged(self):
        for scene,damping,gain in [('stand',0,1),('tennis',.03,5)]:
            model,_=make_model(scene)
            for aid in range(model.nu):
                if 'hand_' not in model.actuator(aid).name:continue
                joint=model.actuator_trnid[aid,0]
                self.assertFalse(model.actuator_forcelimited[aid])
                self.assertEqual(model.actuator_gainprm[aid,0],gain)
                self.assertEqual(model.dof_damping[model.jnt_dofadr[joint]],damping)


if __name__=='__main__':unittest.main()

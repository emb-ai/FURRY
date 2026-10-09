"""Exercise input adaptation against the actual pinned hello_xr transforms."""
import ast
from pathlib import Path
import unittest

from scripts.menu_input_patch import adapt_menu_input


ROOT = Path(__file__).resolve().parents[1]


def prepared_openxr_source():
    """Run only pure OpenXR text transforms; do not export scenes or assets."""
    source = ROOT / 'vendor/OpenXR-SDK-Source/src/tests/hello_xr/openxr_program.cpp'
    if not source.exists():
        raise unittest.SkipTest('Pinned OpenXR source is not available')
    tree = ast.parse((ROOT / 'scripts/prepare_android.py').read_text())
    block = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == 'name'
                and any(isinstance(value, ast.Constant) and value.value == 'openxr_program.cpp'
                        for value in node.test.comparators)):
            block = node
            break
    if block is None:
        raise AssertionError('Could not find prepare_android.py OpenXR transform block')
    # The exporter invokes this helper after the original transforms. Exclude
    # that invocation so this test supplies exactly the helper's input.
    body = [node for node in block.body if not (
        isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name) and node.value.func.id == 'adapt_menu_input')]
    scope = {'text': source.read_text()}
    exec(compile(ast.Module(body=body, type_ignores=[]), 'OpenXR text transforms', 'exec'), scope)
    return scope['text']


class MenuInputPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = prepared_openxr_source()
        cls.after = adapt_menu_input(cls.before)

    def test_only_fast_A_B_Y_and_menu_bindings_remain(self):
        for path in ('/user/hand/right/input/a/click', '/user/hand/right/input/b/click',
                     '/user/hand/left/input/y/click', '/user/hand/left/input/menu/click'):
            self.assertIn(path, self.after)
        for action in ('G1Calibrate();', 'G1ToggleTracking();', 'G1ToggleRecording();', 'G1ToggleMenu();'):
            self.assertIn(action, self.after)
        for removed in ('resetAction', 'viewAction', 'captureModeAction', 'captureLatch',
                        '/input/x/click', '/input/thumbstick/click', 'G1Reset();', 'G1CycleCaptureMode();'):
            self.assertNotIn(removed, self.after)

    def test_aim_and_trigger_use_both_subaction_paths(self):
        for side in ('LEFT', 'RIGHT'):
            self.assertIn(f'bindings.push_back({{m_input.menuAimAction,menuAimPath[Side::{side}]}});', self.after)
            self.assertIn(f'bindings.push_back({{m_input.menuTriggerAction,triggerValuePath[Side::{side}]}});', self.after)
        for hand in ('left', 'right'):
            self.assertIn(f'/user/hand/{hand}/input/aim/pose', self.after)
            self.assertIn(f'/user/hand/{hand}/input/trigger/value', self.after)
        aim_create = self.after.index('strcpy_s(actionInfo.actionName, "menu_aim")')
        subactions = self.after.rfind('actionInfo.subactionPaths = m_input.handSubactionPath.data();', 0, aim_create)
        self.assertGreater(subactions, self.after.index('&m_input.recordAction'))
        trigger_create = self.after.index('strcpy_s(actionInfo.actionName, "menu_trigger")')
        self.assertIn('actionInfo.actionType = XR_ACTION_TYPE_FLOAT_INPUT;', self.after[aim_create:trigger_create])

    def test_two_aim_spaces_are_created_and_destroyed_with_action_set(self):
        for side in ('LEFT', 'RIGHT'):
            self.assertEqual(self.after.count(f'xrCreateActionSpace(m_session, &actionSpaceInfo, &m_input.menuAimSpace[Side::{side}])'), 1)
        destructor = self.after[self.after.index('~OpenXrProgram()'):self.after.index('static void LogLayersAndExtensions')]
        self.assertLess(destructor.index('xrDestroySpace(m_input.menuAimSpace[hand])'), destructor.index('xrDestroyActionSet'))
        self.assertIn('menuAimSpace{};', self.after)

    def test_ray_poses_use_predicted_time_and_stage_before_preparing_menu(self):
        start = self.after.index('        G1SubmitTracking(tracking);')
        end = self.after.index('        G1PrepareFrame();', start)
        frame = self.after[start:end]
        self.assertIn('xrLocateSpace(m_input.menuAimSpace[hand],m_appSpace,predictedDisplayTime,&location)', frame)
        self.assertIn('G1SubmitMenuRay(hand,trackedPose(location.pose),valid,', frame)
        self.assertIn('(location.locationFlags&required)==required', frame)
        self.assertIn('m_input.menuAimActive[hand] && IsSessionFocused()', frame)
        self.assertIn('G1SubmitMenuRay(hand,TrackedPose{},false,false,0.f)', self.after)

    def test_focus_loss_clears_trigger_and_guards_button_dispatch(self):
        self.assertIn('m_input.menuTrigger[hand].isActive = XR_FALSE;', self.after)
        self.assertIn('m_input.menuTrigger[hand].currentState = 0.f;', self.after)
        self.assertIn('if(press && focused)', self.after)
        self.assertIn('if (IsSessionFocused() && (quitValue.isActive', self.after)
        self.assertIn('trigger.isActive && IsSessionFocused(),trigger.currentState', self.after)

    def test_passthrough_follows_runtime_setting(self):
        self.assertIn('if(G1PassthroughVisible())layers.push_back(passthrough.Layer());', self.after)

    def test_source_drift_fails_instead_of_partially_patching(self):
        with self.assertRaisesRegex(ValueError, 'record latch: expected 1 occurrence, found 0'):
            adapt_menu_input(self.before.replace('ButtonLatch recordLatch, captureLatch;', 'ButtonLatch changed;'))
        with self.assertRaisesRegex(ValueError, 'expected 1 occurrence, found 2'):
            adapt_menu_input(self.before + '\nButtonLatch recordLatch, captureLatch;')
        with self.assertRaisesRegex(ValueError, 'record latch'):
            adapt_menu_input(self.after)


if __name__ == '__main__':
    unittest.main()

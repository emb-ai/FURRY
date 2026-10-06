"""Export the verified scene and adapt the pinned Khronos hello_xr lifecycle."""
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
import struct
import json
import hashlib
from collections import Counter
import numpy as np
import fast_simplification
import mujoco
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g1_sim.scene import make_model
from g1_sim.collisions import primitive_collisions
from g1_sim.controller import UPSTREAM

assets = ROOT / 'android/assets'
assets.mkdir(parents=True, exist_ok=True)
model, xml = make_model('lab', hands=True)
xml, collision_report = primitive_collisions(xml)
(assets/'collision_budget.json').write_text(json.dumps(collision_report, indent=2)+'\n')
root = ET.fromstring(xml)
root.find('compiler').set('meshdir', 'meshes')
# Hide duplicate collision geometry in the GLES renderer, retain physical shape.
pelvis = root.find(".//body[@name='pelvis']")
for geom in pelvis.iter('geom'):
    if geom.get('contype') != '0':
        geom.set('group', '3')
ET.ElementTree(root).write(assets/'scene.xml', encoding='unicode')
shutil.copy2(UPSTREAM/'assets/ckpts/twist2_1017_20k.onnx', assets/'policy.onnx')
meshdir = assets/'meshes'
meshdir.mkdir(exist_ok=True)
for mesh in root.iter('mesh'):
    shutil.copy2(UPSTREAM/'assets/g1/meshes'/mesh.get('file'), meshdir/mesh.get('file'))
shutil.copy2(ROOT/'THIRD_PARTY_NOTICES.txt', assets/'TWIST2_LICENSE.txt')
shutil.copy2(ROOT/'vendor/mujoco/LICENSE', assets/'MUJOCO_LICENSE.txt')
shutil.copy2(ROOT/'vendor/OpenXR-SDK-Source/LICENSE', assets/'OPENXR_LICENSE.txt')
shutil.copy2(ROOT/'vendor/OpenXR-SDK-Source/LICENSES/Apache-2.0.txt', assets/'OPENXR_APACHE_2.txt')
shutil.copy2(ROOT/'vendor/onnxruntime-android/LICENSE', assets/'ORT_LICENSE.txt')
shutil.copy2(ROOT/'vendor/onnxruntime-android/ThirdPartyNotices.txt', assets/'ORT_THIRD_PARTY.txt')
model = mujoco.MjModel.from_xml_path(str(assets/'scene.xml'))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
scene = mujoco.MjvScene(model, maxgeom=2048)
option = mujoco.MjvOption()
option.geomgroup[3] = 0
mujoco.mjv_updateScene(model, data, option, None, mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scene)
mesh_usage = Counter(g.dataid//2 for g in scene.geoms[:scene.ngeom] if g.type == mujoco.mjtGeom.mjGEOM_MESH)
# Match the GLES primitive tessellation, including both caps of cylinders.
primitive_faces = {mujoco.mjtGeom.mjGEOM_BOX:12, mujoco.mjtGeom.mjGEOM_PLANE:12,
                   mujoco.mjtGeom.mjGEOM_SPHERE:256, mujoco.mjtGeom.mjGEOM_ELLIPSOID:256,
                   mujoco.mjtGeom.mjGEOM_CYLINDER:64, mujoco.mjtGeom.mjGEOM_CAPSULE:64}
primitive_triangles = 2 + sum(primitive_faces.get(g.type, 0) for g in scene.geoms[:scene.ngeom]
                              if g.type != mujoco.mjtGeom.mjGEOM_PLANE)
targets = [min(int(n), max(200, min(1800, int(n*.12)))) for n in model.mesh_facenum]
weighted = sum(targets[i]*uses for i, uses in mesh_usage.items())
ratio = min(1., (90000-primitive_triangles)/max(weighted,1))
targets = [max(4, int(n*ratio)) for n in targets]
# Visual simplification is independent of the primitive collision model.
triangles = 0
mesh_counts = []
with (assets/'visual_meshes.bin').open('wb') as out:
    out.write(struct.pack('<I', model.nmesh))
    for i in range(model.nmesh):
        va, vn = model.mesh_vertadr[i], model.mesh_vertnum[i]
        fa, fn = model.mesh_faceadr[i], model.mesh_facenum[i]
        vertices = model.mesh_vert[va:va+vn].astype(np.float64)
        faces = model.mesh_face[fa:fa+fn]
        target = targets[i]
        if target < fn:
            vertices, faces = fast_simplification.simplify(vertices, faces, target_count=target)
        points = vertices[faces]
        normal = np.cross(points[:,1]-points[:,0], points[:,2]-points[:,0])
        normal /= np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
        packed = np.concatenate([points,np.repeat(normal[:,None,:],3,axis=1)],axis=2).astype('<f4')
        out.write(struct.pack('<I',len(faces)*3))
        out.write(packed.tobytes())
        triangles += len(faces)
        mesh_counts.append(len(faces))
print('Visual mesh triangles:', model.nmeshface, '->',triangles)
scene_triangles = primitive_triangles + sum(mesh_counts[i]*uses for i, uses in mesh_usage.items())
hud_max_triangles = 8192  # bounded text buffer plus raw skeleton inset/labels
assert scene_triangles + hud_max_triangles <= 100000, f'Scene exceeds polygon budget: {scene_triangles}'
budget = dict(maximum_triangles_per_eye=100000, scene_triangles_per_eye=scene_triangles,
              hud_max_triangles=hud_max_triangles, total_max_triangles_per_eye=scene_triangles+hud_max_triangles,
              primitive_triangles=primitive_triangles, unique_visual_mesh_triangles=triangles,
              original_mesh_triangles=int(model.nmeshface), method='quadric edge collapse',
              mesh_instances=sum(mesh_usage.values()), physics_meshes_unchanged=False, robot_collision_meshes=0)
(assets/'mesh_budget.json').write_text(json.dumps(budget, indent=2)+'\n')
print('Complete rendered scene triangles per eye:', scene_triangles, '/ 100000')
font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
title_font = ImageFont.truetype(font_path, 48)
body_font = ImageFont.truetype(font_path, 36)
states = ['G1 · Для калибровки нажмите A', 'G1 · GMR: полный скелет',
                              'G1 · Трекинг на паузе', 'G1 · Нет полного скелета',
                              'G1 · Робот упал. X → рестарт', 'G1 · GMR: большая ошибка позы']
for state in range(18):
    title = states[state % 6]
    panel = Image.new('RGBA', (1024, 512), (18, 28, 43, 255))
    draw = ImageDraw.Draw(panel)
    draw.text((40, 28), title, font=title_font, fill=(107,217,234))
    for y,line in enumerate(['Встаньте прямо, смотрите вперёд',
                            'A → начало координат и калибровка тела',
                            'B → пауза / продолжение трекинга',
                            'Grip слева / справа → сжать свою кисть',
                            'X → рестарт; Y → начать / закончить запись',
                            ['Запись выключена. Сначала Y, затем A.', '● ИДЁТ ЗАПИСЬ · Y → сохранить', 'Ошибка записи: проверьте свободное место'][state // 6]]):
        draw.text((40,125+y*58),line,font=body_font,fill=(228,236,245))
    (assets/f'controls_{state}.rgba').write_bytes(panel.tobytes())
(assets/'controls.rgba').unlink(missing_ok=True)
# Fixed native HUD glyph atlas. Runtime only changes a small vertex buffer.
font = ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', 26)
atlas = Image.new('L', (512, 240), 0)
draw = ImageDraw.Draw(atlas)
for code in range(32, 127):
    i=code-32;draw.text(((i%16)*32+2, (i//16)*40+2), chr(code), font=font, fill=255)
draw.rectangle((15*32,5*40,512,240),fill=255)
(assets/'stats_font.bin').write_bytes(atlas.tobytes())


src = ROOT/'vendor/OpenXR-SDK-Source/src/tests/hello_xr'
dest = ROOT/'android/generated/xr'
dest.mkdir(parents=True, exist_ok=True)
names = ['pch.h','common.h','check.h','logger.h','logger.cpp','options.h',
         'platformdata.h','platformplugin.h','platformplugin_factory.cpp','platformplugin_android.cpp',
         'graphicsplugin.h','graphicsplugin_factory.cpp','openxr_program.h','openxr_program.cpp','main.cpp']
for name in names:
    text = (src/name).read_text()
    if name == 'pch.h':
        text = text.replace('#include <glad/egl.h>', '#include <EGL/egl.h>\n#include <GLES3/gl32.h>')
    if name == 'options.h':
        text = text.replace('AppSpace{"Local"}', 'AppSpace{"Stage"}')
    if name == 'main.cpp':
        text = text.replace('#include "pch.h"', '#include "pch.h"\n#include "quest_runtime.h"\n#include <exception>')
        text = text.replace('ALooper_pollAll(', 'ALooper_pollOnce(')
        text = text.replace('program->CreateSwapchains();', '''program->CreateSwapchains();
        // Stop the worker before XR/graphics teardown, including exception paths.
        struct RuntimeGuard { ~RuntimeGuard(){ G1Shutdown(std::uncaught_exceptions() > 0); } } runtimeGuard;
        G1Initialize(app);''')
        text = text.replace('program->PollActions();', 'G1SetActive(program->IsSessionFocused());\n            program->PollActions();')
        text = text.replace('if (!program->IsSessionRunning()) {', 'if (!program->IsSessionRunning()) {\n                G1SetActive(false);')
    if name == 'openxr_program.cpp':
        text = text.replace('#include "pch.h"', '#include "pch.h"\n#include "quest_runtime.h"\n#include "passthrough.h"\n#include "body_tracking.h"')
        text = text.replace('struct OpenXrProgram : IOpenXrProgram {',
                            'struct OpenXrProgram : IOpenXrProgram {\n    Passthrough passthrough; BodyTracking bodyTracking; uint64_t trackingSequence=0;')
        text = text.replace('XrAction quitAction{XR_NULL_HANDLE};',
                            'XrAction quitAction{XR_NULL_HANDLE};\n        XrAction calibrateAction{XR_NULL_HANDLE}, pauseAction{XR_NULL_HANDLE}, resetAction{XR_NULL_HANDLE}, recordAction{XR_NULL_HANDLE};')
        text = text.replace('CHECK_XRCMD(xrCreateAction(m_input.actionSet, &actionInfo, &m_input.quitAction));', '''CHECK_XRCMD(xrCreateAction(m_input.actionSet, &actionInfo, &m_input.quitAction));
            strcpy_s(actionInfo.actionName, "calibrate_tracking");
            strcpy_s(actionInfo.localizedActionName, "Calibrate tracking");
            CHECK_XRCMD(xrCreateAction(m_input.actionSet, &actionInfo, &m_input.calibrateAction));
            strcpy_s(actionInfo.actionName, "pause_tracking");
            strcpy_s(actionInfo.localizedActionName, "Pause tracking");
            CHECK_XRCMD(xrCreateAction(m_input.actionSet, &actionInfo, &m_input.pauseAction));
            strcpy_s(actionInfo.actionName, "restart_simulation");
            strcpy_s(actionInfo.localizedActionName, "Restart simulation");
            CHECK_XRCMD(xrCreateAction(m_input.actionSet, &actionInfo, &m_input.resetAction));
            strcpy_s(actionInfo.actionName, "record_episode");
            strcpy_s(actionInfo.localizedActionName, "Record episode");
            CHECK_XRCMD(xrCreateAction(m_input.actionSet, &actionInfo, &m_input.recordAction));''')
        text = text.replace('suggestedBindings.interactionProfile = oculusTouchInteractionProfilePath;', '''suggestedBindings.interactionProfile = oculusTouchInteractionProfilePath;
            XrPath aButton, bButton, xButton, yButton;
            CHECK_XRCMD(xrStringToPath(m_instance,"/user/hand/right/input/a/click",&aButton));
            CHECK_XRCMD(xrStringToPath(m_instance,"/user/hand/right/input/b/click",&bButton));
            bindings.push_back({m_input.calibrateAction,aButton});
            bindings.push_back({m_input.pauseAction,bButton});
            CHECK_XRCMD(xrStringToPath(m_instance,"/user/hand/left/input/x/click",&xButton));
            CHECK_XRCMD(xrStringToPath(m_instance,"/user/hand/left/input/y/click",&yButton));
            bindings.push_back({m_input.resetAction,xButton});
            bindings.push_back({m_input.recordAction,yButton});''')
        text = text.replace('// There were no subaction paths specified for the quit action,', '''for(int button=0;button<4;button++){
            XrActionStateGetInfo info{XR_TYPE_ACTION_STATE_GET_INFO};
            const XrAction buttons[]={m_input.calibrateAction,m_input.pauseAction,m_input.resetAction,m_input.recordAction};
            info.action=buttons[button];
            XrActionStateBoolean state{XR_TYPE_ACTION_STATE_BOOLEAN};
            CHECK_XRCMD(xrGetActionStateBoolean(m_session,&info,&state));
            if(state.isActive && state.changedSinceLastSync && state.currentState){
                if(button==0)G1Calibrate();else if(button==1)G1ToggleTracking();
                else if(button==2)G1Reset();else G1ToggleRecording();
            }
        }
        // There were no subaction paths specified for the quit action,''')
        text = text.replace('~OpenXrProgram() override {', '~OpenXrProgram() override {\n        bodyTracking.Shutdown(); passthrough.Shutdown();')
        text = text.replace('case XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING:', '''case XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING: {
                    const auto& change = *reinterpret_cast<const XrEventDataReferenceSpaceChangePending*>(event);
                    if(change.session == m_session && change.referenceSpaceType == XR_REFERENCE_SPACE_TYPE_STAGE)
                        G1ReferenceSpaceChange(change.changeTime);
                    break;
                }''')
        text = text.replace('std::vector<const char*> extensions;',
                            'std::vector<const char*> extensions{XR_FB_PASSTHROUGH_EXTENSION_NAME}; bodyTracking.enabled=BodyTracking::Enable(extensions);')
        text = text.replace('CHECK_XRCMD(xrCreateSession(m_instance, &createInfo, &m_session));',
                            'CHECK_XRCMD(xrCreateSession(m_instance, &createInfo, &m_session));\n            passthrough.Initialize(m_instance, m_session); bodyTracking.Initialize(m_instance, m_systemId, m_session);')
        text = text.replace('if (frameState.shouldRender == XR_TRUE) {',
                            'if (frameState.shouldRender == XR_TRUE) {\n            layers.push_back(passthrough.Layer());')
        a = text.index('        layer.layerFlags =')
        b = text.index('        layer.viewCount', a)
        text = text[:a] + '        layer.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT | XR_COMPOSITION_LAYER_UNPREMULTIPLIED_ALPHA_BIT;\n' + text[b:]
        text = text.replace('"HelloXR"', '"G1 Quest Lab"')
        text = text.replace('CHECK_XRCMD(xrRequestExitSession(m_session));', 'G1Reset();')
        text = text.replace('"Quit Session"', '"Reset scene"').replace('"Grab Object"', '"Close fingers"')
        # Sample haptics do not represent physical contacts; omit that feedback.
        a = text.index('                if (grabValue.currentState > 0.9f) {')
        b = text.index('\n            }', a)
        text = text[:a] + text[b:]
        text = text.replace('swapchainCreateInfo.width = vp.recommendedImageRectWidth;',
                            'swapchainCreateInfo.width = std::min(vp.recommendedImageRectWidth, 1440u);')
        text = text.replace('swapchainCreateInfo.height = vp.recommendedImageRectHeight;',
                            'swapchainCreateInfo.height = uint32_t(float(vp.recommendedImageRectHeight) * swapchainCreateInfo.width / vp.recommendedImageRectWidth);')
        text = text.replace('m_input.handScale[hand] = 1.0f - 0.5f * grabValue.currentState;',
                            'm_input.handScale[hand] = 1.0f - 0.5f * grabValue.currentState;\n                G1Grip(static_cast<int>(hand), grabValue.currentState);')
        # Remove sample space/hand cubes. Simulation draws the real robot meshes.
        text = text.replace('            return false;  // There is no valid tracking poses for the views.',
                            '            TrackingFrame missing;missing.sequence=++trackingSequence;missing.xr_time_ns=predictedDisplayTime;G1SubmitTracking(missing);\n            return false;  // There is no valid tracking poses for the views.')
        a = text.index('        // For each locatable space that we want to visualize')
        b = text.index('        // Render view to the appropriate part', a)
        tracking = '''        std::vector<Cube> cubes;
        auto trackedPose=[](const XrPosef& p){TrackedPose t;
            t.position={p.position.x,p.position.y,p.position.z};
            t.quaternion={p.orientation.w,p.orientation.x,p.orientation.y,p.orientation.z};return t;};
        TrackingFrame tracking;
        tracking.head=trackedPose(m_views[0].pose);
        for(int a=0;a<3;a++)tracking.head.position[a]=(tracking.head.position[a]+trackedPose(m_views[1].pose).position[a])*.5;
        tracking.xr_time_ns=predictedDisplayTime;
        tracking.sequence=++trackingSequence;
        // View-state bit values differ from space-location bits; convert explicitly.
        auto vf=viewState.viewStateFlags;
        tracking.location_flags[0]=((vf&XR_VIEW_STATE_ORIENTATION_VALID_BIT)?XR_SPACE_LOCATION_ORIENTATION_VALID_BIT:0)
            |((vf&XR_VIEW_STATE_POSITION_VALID_BIT)?XR_SPACE_LOCATION_POSITION_VALID_BIT:0)
            |((vf&XR_VIEW_STATE_ORIENTATION_TRACKED_BIT)?XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT:0)
            |((vf&XR_VIEW_STATE_POSITION_TRACKED_BIT)?XR_SPACE_LOCATION_POSITION_TRACKED_BIT:0);
        tracking.valid=true;
        for(int hand=0;hand<2;hand++){
            XrSpaceLocation location{XR_TYPE_SPACE_LOCATION};
            XrResult r=xrLocateSpace(m_input.handSpace[hand],m_appSpace,predictedDisplayTime,&location);
            const auto required=XR_SPACE_LOCATION_POSITION_VALID_BIT|XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
            if(XR_FAILED(r) || (location.locationFlags&required)!=required || !m_input.handActive[hand])tracking.valid=false;
            tracking.location_flags[hand+1]=XR_SUCCEEDED(r)?location.locationFlags:0;
            tracking.hand_active[hand]=m_input.handActive[hand];
            tracking.hands[hand]=trackedPose(location.pose);
        }
        bodyTracking.Sample(m_appSpace,predictedDisplayTime,tracking);
        G1SubmitTracking(tracking);
        G1PrepareFrame();

'''
        text = text[:a] + tracking + text[b:]
    (dest/name).write_text(text)
print('Android scene and OpenXR lifecycle prepared:',assets)

# Identify the exact scene, meshes, policy and retargeter used by each episode.
hashes = {}
for path in sorted([assets/'scene.xml', assets/'policy.onnx', *meshdir.glob('*'),
                    ROOT/'android/native/retarget.cpp', ROOT/'android/native/simulation.cpp']):
    hashes[str(path.relative_to(assets) if path.is_relative_to(assets) else path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
metadata = dict(schema_version=1, retargeting='native_gmr_meta_full_body', input_mode='meta_full_body_hmd_touch_grip', pose_frame='OpenXR STAGE; metres; quaternion wxyz',
                head_pose='mean stereo-eye position; left-eye orientation',
                physics_hz=1000, policy_hz=100, nq=model.nq, nv=model.nv, nu=model.nu,
                joints=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)],
                source_twist2='b06178f19a22f2138cbd31f60c6d494bc263f67d',
                source_gmr='bb1bbe40774794fceb2a7c579a3464a28e68c844', sha256=hashes,
                state_phase='before next 10 physics steps; ctrl is last physics-step output',
                clocks='xr_time_ns is predicted display time; receive_ns is steady_clock, no clock-offset fit',
                missing_data='valid and per-pose OpenXR flags are authoritative; invalid numerical poses must not be used',
                limitations=['no measured elbows, legs or finger skeleton', 'no video', 'no complete controller hidden state for dynamic restart'])
(assets/'recording_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')

from export_gmr import export as export_gmr
export_gmr()

for file in ['gmr_model.xml','gmr_config.txt']:
    metadata['sha256'][file]=hashlib.sha256((assets/file).read_bytes()).hexdigest()
for file in ['gmr.cpp','gmr.h','meta_retarget.cpp','meta_retarget.h','body_tracking.h','quest_runtime.cpp','runtime_stats.h','stats_hud.h','recording_export.h']:
    metadata['sha256']['android/native/'+file]=hashlib.sha256((ROOT/'android/native'/file).read_bytes()).hexdigest()
metadata['ik_error_semantics']='unweighted GMR stage-2 SE3 residual norm; mixed metres/radians, not wrist distance'
metadata['body_tracking']={'source':'XR_FB_body_tracking + XR_META_body_tracking_full_body','lower_body':'runtime-estimated, not measured foot trackers','retargeting':'GMR two-stage SE3 box QP; Meta bind-skeleton adapter','root_xy':'Meta pelvis displacement from A, scaled to robot proportions; head-relative sway excluded','scaling':'leg height and arm lengths from Meta bind skeleton; source-specific bone-axis offsets','joints':['Pelvis','Spine3','Left_Hip','Right_Hip','Left_Knee','Right_Knee','Left_Foot','Right_Foot','Left_Shoulder','Right_Shoulder','Left_Elbow','Right_Elbow','Left_Wrist','Right_Wrist']}
metadata['limitations']=['Meta lower-body poses are estimates','no video','no complete policy hidden state for dynamic restart','Meta source adapter requires hardware validation']
(assets/'recording_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
for name in ['GMR','MINK']:
    shutil.copy2(ROOT/'third_party'/f'{name}_LICENSE.txt',assets/f'{name}_LICENSE.txt')

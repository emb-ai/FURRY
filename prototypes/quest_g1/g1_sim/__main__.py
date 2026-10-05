import argparse
from collections import deque
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .controller import Controller, ROOT, standing_command
from .scene import make_model


def arm_command(t):
    """Small smooth reference movements; the learned policy keeps balance."""
    command = standing_command()
    wave = .5 * (1-np.cos(2*np.pi*t/6))
    command[6+15] -= .35*wave
    command[6+22] -= .35*wave
    command[6+18] += .3*wave
    command[6+25] += .3*wave
    return command


def configure_camera(cam):
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [.4, 0, .65]
    cam.distance = 3.7
    cam.azimuth = 135
    cam.elevation = -22


def snapshot(model, data, path, ego=False):
    from PIL import Image
    cam = mujoco.MjvCamera()
    configure_camera(cam)
    with mujoco.Renderer(model, height=800, width=1280) as renderer:
        renderer.update_scene(data, camera="ego" if ego else cam)
        Image.fromarray(renderer.render()).save(path)


def replay(args):
    """Visual state playback, not a new physics rollout of recorded actions."""
    episode = np.load(args.replay, allow_pickle=False)
    model = mujoco.MjModel.from_xml_string(str(episode["scene_xml"]))
    data = mujoco.MjData(model)
    times, positions, velocities = episode["time"], episode["qpos"], episode["qvel"]
    if len(times) == 0 or positions.shape != (len(times), model.nq) or velocities.shape != (len(times), model.nv):
        raise ValueError("Invalid episode state dimensions")
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all() or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Episode must contain finite states with increasing timestamps")
    viewer = None
    if not args.headless:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
        with viewer.lock():
            configure_camera(viewer.cam)
    start = time.perf_counter()
    next_frame = 0.
    for i, t in enumerate(times):
        if viewer and not viewer.is_running():
            break
        data.time = t
        data.qpos[:] = positions[i]
        data.qvel[:] = velocities[i]
        data.ctrl[:] = episode["ctrl"][i]
        mujoco.mj_forward(model, data)
        if viewer and t >= next_frame:
            viewer.sync()
            next_frame = t + 1/60
            if not args.fast:
                time.sleep(max(0, t-times[0]-(time.perf_counter()-start)))
    if viewer:
        viewer.close()
    if args.screenshot:
        snapshot(model, data, args.screenshot, args.ego)
    print(f"Replayed {i+1} recorded states; final t={data.time:.3f}s", flush=True)


def run(args):
    model, xml = make_model(args.scene, hands=not args.fixed_hands)
    data = mujoco.MjData(model)
    controller = Controller(model)
    controller.initialize(data)
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    queued_keys = deque()
    paused, ego, grip = False, args.ego, 0.
    demo_start = 0. if args.demo else None
    demo_stop = False
    viewer = None
    recording = []
    record_path = Path(args.record).resolve() if args.record else None
    min_height, max_tilt = float(data.qpos[2]), 0.
    wall_start = time.perf_counter()
    pace_start = wall_start
    pace_sim = 0.
    frame_next = 0.
    status_next = 5.
    print(f"Unitree G1 / TWIST2 CPU | scene={args.scene} | joints={model.nu}", flush=True)
    print("F8 reset | F9 pause | F10 arm demo | F11 hands | F12 camera | close window to exit", flush=True)
    if not args.headless:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data, key_callback=queued_keys.append,
                                              show_left_ui=False, show_right_ui=False)
        with viewer.lock():
            configure_camera(viewer.cam)
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
            if ego:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = model.camera("ego").id
    try:
        while (viewer is None or viewer.is_running()) and (args.seconds <= 0 or data.time < args.seconds):
            while queued_keys:
                key = queued_keys.popleft()
                if key == 297:  # GLFW_KEY_F8
                    if record_path:
                        print("Finish this recording before resetting (close and restart).", flush=True)
                        continue
                    controller.initialize(data)
                    demo_start = None
                    demo_stop = False
                    grip = 0.
                    frame_next, status_next = 0., 5.
                    min_height, max_tilt = float(data.qpos[2]), 0.
                    pace_start, pace_sim = time.perf_counter(), data.time
                elif key == 298:  # F9
                    paused = not paused
                    print(f"{'Paused' if paused else 'Running'} at t={data.time:.3f}", flush=True)
                    pace_start, pace_sim = time.perf_counter(), data.time
                elif key == 299:  # F10
                    if demo_start is None:
                        demo_start = data.time
                        demo_stop = False
                    else:
                        demo_stop = True  # Finish the cycle for a smooth return.
                elif key == 300:  # F11
                    grip = 1.-grip
                    print(f"Hands: {'closed' if grip else 'open'}", flush=True)
                elif key == 301 and viewer:  # F12
                    ego = not ego
                    with viewer.lock():
                        if ego:
                            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                            viewer.cam.fixedcamid = model.camera("ego").id
                        else:
                            configure_camera(viewer.cam)
            if paused:
                if viewer:
                    viewer.sync()
                time.sleep(.016)
                continue
            if demo_stop and demo_start is not None and (data.time-demo_start) % 6 < model.opt.timestep*1.5:
                demo_start = None
                demo_stop = False
            command = arm_command(data.time-demo_start) if demo_start is not None else standing_command()
            controller.step(data, command, grip)
            min_height = min(min_height, float(data.qpos[2]))
            max_tilt = max(max_tilt, float(np.arccos(np.clip(data.xmat[model.body('pelvis').id, 8], -1, 1))))
            if not np.isfinite(data.qpos).all() or data.qpos[2] < .35:
                raise RuntimeError(f"Robot fell or simulation became unstable at t={data.time:.3f}")
            if record_path and controller.step_index % controller.period == 0:
                recording.append((data.time, data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), command.copy(), grip))
            if viewer and data.time >= frame_next:
                viewer.sync()
                frame_next += 1/60
                if not args.fast:
                    delay = (data.time-pace_sim) - (time.perf_counter()-pace_start)
                    if delay > 0:
                        time.sleep(delay)
                    elif delay < -.25:
                        # Do not accumulate unlimited catch-up after dragging the window.
                        pace_start, pace_sim = time.perf_counter(), data.time
            if viewer and data.time >= status_next:
                print(f"t={data.time:6.1f}s | pelvis={data.qpos[2]:.3f}m | policy p95={np.percentile(controller.inference_ms[-500:],95):.2f}ms", flush=True)
                status_next += 5
    finally:
        elapsed = time.perf_counter()-wall_start
        stats = {"scene":args.scene, "hands":not args.fixed_hands, "sim_seconds":float(data.time),
                 "wall_seconds":elapsed, "realtime_factor":float(data.time)/max(elapsed,1e-6),
                 "minimum_pelvis_height_m":min_height, "maximum_pelvis_tilt_deg":float(np.degrees(max_tilt)),
                 "final_pelvis_xyz":data.qpos[:3].tolist(),
                 "policy_inference_p95_ms":float(np.percentile(controller.inference_ms,95)) if controller.inference_ms else 0.,
                 "physics_hz":1/model.opt.timestep, "policy_hz":1/(model.opt.timestep*controller.period),
                 "mujoco_warnings":{str(i):int(w.number) for i,w in enumerate(data.warning) if w.number}}
        if record_path and recording:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(record_path, time=[r[0] for r in recording], qpos=[r[1] for r in recording],
                                qvel=[r[2] for r in recording], ctrl=[r[3] for r in recording],
                                command=[r[4] for r in recording], grip=[r[5] for r in recording],
                                scene_xml=xml, scene=args.scene, metadata=json.dumps(stats))
            print(f"Recording saved: {record_path}", flush=True)
        if viewer:
            viewer.close()
        print(json.dumps(stats, indent=2), flush=True)
        if args.report:
            Path(args.report).write_text(json.dumps(stats, indent=2)+"\n")
    if args.screenshot:
        snapshot(model, data, args.screenshot, args.ego)


def main():
    parser = argparse.ArgumentParser(description="Unitree G1 with TWIST2 on MuJoCo / macOS")
    parser.add_argument("--scene", choices=["stand", "cup", "push_t", "lab"], default="lab")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fixed-hands", action="store_true", help="Use original 29-DoF model without articulated fingers")
    parser.add_argument("--seconds", type=float, default=0)
    parser.add_argument("--demo", action="store_true", help="Smooth arm reference demonstration")
    parser.add_argument("--ego", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--record", help="Save states and commands to an NPZ episode")
    parser.add_argument("--replay", help="Visual playback of recorded NPZ states (no policy or physics stepping)")
    parser.add_argument("--report", help="Write benchmark statistics as JSON")
    parser.add_argument("--screenshot", help="Save final rendered frame as PNG")
    args = parser.parse_args()
    if args.headless and args.seconds <= 0 and not args.replay:
        parser.error("--headless requires a positive --seconds")
    if args.record and not args.record.endswith(".npz"):
        parser.error("--record filename must end with .npz")
    if args.replay:
        replay(args)
    else:
        run(args)


if __name__ == "__main__":
    main()

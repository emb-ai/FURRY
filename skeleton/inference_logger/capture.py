"""RealSense raw recorder + latest-frame inference, independent of Quest LAN."""
import json
from pathlib import Path
import shutil
import threading
import time
import uuid

import cv2
import numpy as np

from .storage import Session, save_json
from .transport import QuestServer

JOINTS = {'lhip': 11, 'lknee': 13, 'lankle': 15, 'rhip': 12,
          'rknee': 14, 'rankle': 16, 'nose': 0, 'lshoulder': 5, 'rshoulder': 6}


class Inference:
    def __init__(self, model_path, tag_layout=None):
        from ultralytics import YOLO
        if not Path(model_path).exists():
            raise FileNotFoundError(model_path)
        self.model = YOLO(str(model_path), task='pose')
        self.layout = json.loads(Path(tag_layout).read_text(encoding='utf-8')) if tag_layout else None
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), params)

    def process(self, image, depth_m, intr):
        import pyrealsense2 as rs
        from fusion import window_median, line_median, person_distance, PARENT
        from .tag_geometry import tag_pose
        begin = time.monotonic_ns()
        result = self.model.predict(image, imgsz=640, conf=.3, classes=0, verbose=False)[0]
        people = []
        if result.keypoints is not None:
            xy = result.keypoints.xy.cpu().numpy()
            confidence = result.keypoints.conf.cpu().numpy()
            for p, c in zip(xy, confidence):
                people.append({'xy_px': p.tolist(), 'confidence': c.tolist()})
        # No silent identity switch in a multi-person image.
        selected = 0 if len(people) == 1 else None
        joints = {}
        if selected is not None:
            person = people[selected]
            points = {i: p for i, (p, c) in enumerate(zip(person['xy_px'], person['confidence'])) if c > .5}
            distance = person_distance(depth_m, points)
            for i, p in points.items():
                d = window_median(depth_m, *p, distance)
                source = 'window'
                if d is None and PARENT.get(i) in points:
                    d = line_median(depth_m, *points[PARENT[i]], *p, distance)
                    source = 'line'
                if d is not None:
                    joints[str(i)] = {'p': rs.rs2_deproject_pixel_to_point(intr, p, d),
                        'conf': person['confidence'][i], 'src': source,
                        'provenance': 'measured_depth_surface' if source == 'window' else 'inferred'}
        corners, ids, _ = self.detector.detectMarkers(image)
        tags = []
        k = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.]])
        for q, marker in zip(corners, [] if ids is None else ids.ravel()):
            q = q.reshape(4, 2)
            item = {'tag_id': int(marker), 'family': '36h11', 'corners_px': q.tolist(),
                    'corner_order': 'marker top-left, top-right, bottom-right, bottom-left',
                    'size_m': self.layout['marker_m'] if self.layout else None,
                    'tag_to_camera': None, 'quality': 'pose_unavailable'}
            if self.layout:
                pose = tag_pose(q, self.layout['marker_m'], k, np.asarray(intr.coeffs))
                if pose is not None:
                    r, t = pose
                    item.update(tag_to_camera={'R': r.tolist(), 't_m': t.tolist()},
                                quality='passed_existing_reprojection_and_ambiguity_gate')
            tags.append(item)
        return {'people_2d': people, 'selected_person': selected, 'joints': joints,
                'tags': tags, 'coordinate_frame': 'camera_color_optical',
                'axes': 'x right, y down, z forward', 'units': 'm',
                'smoothing': 'none', 'tag_skeleton_fusion': 'not_applied',
                'inference_start_ns': str(begin), 'inference_end_ns': str(time.monotonic_ns())}


def packet(result, timestamp_ms, seq, frame_id, session_id):
    missing = {'p': [0., 0., 0.], 'conf': 0., 'src': 'missing'}
    joints = {name: {k: result['joints'].get(str(i), missing)[k] for k in ('p', 'conf', 'src')}
              for name, i in JOINTS.items()}
    left, right = joints['lhip'], joints['rhip']
    pelvis = dict(missing)
    if left['conf'] > 0 and right['conf'] > 0:
        pelvis = {'p': [(a+b)/2 for a, b in zip(left['p'], right['p'])],
                  'conf': min(left['conf'], right['conf']), 'src': 'window'}
    return {'t': timestamp_ms, 'seq': seq, 'frame': 'camera', 'pelvis': pelvis,
            'joints': joints, 'backend': 'inference_logger_rgbd', 'fps': 0.,
            'frame_id': frame_id, 'session_id': session_id}


def record(args):
    import pyrealsense2 as rs
    infer = Inference(args.model, args.tag_layout)
    session = Session(args.output, {'source': 'RealSense', 'opencv': cv2.__version__,
        'camera_id': args.camera_id, 'calibration_id': 'factory_intrinsics',
        'capture_time_semantics': 'host acquisition; not exposure',
        'quest_pose_frame': 'OpenXR STAGE', 'spatial_registration': 'not_applied'})
    pipeline = rs.pipeline()
    stop = threading.Event()
    condition = threading.Condition()
    slot = [None]
    errors = []
    capture_thread = None
    server = None
    started = False
    failure = None
    run_id = str(uuid.uuid4())
    try:
        bundle = session.path / 'bundle'
        bundle.mkdir()
        model = Path(args.model)
        if model.is_dir():
            shutil.copytree(model, bundle / 'model')
        else:
            shutil.copy2(model, bundle / model.name)
        if args.tag_layout:
            shutil.copy2(args.tag_layout, bundle / 'tag_layout.json')
        # Freeze code dependencies as provenance, without changing source files.
        for name in ('fusion.py',):
            shutil.copy2(Path(__file__).resolve().parents[1] / name, bundle / name)
        code = bundle / 'inference_logger'
        code.mkdir()
        for source in Path(__file__).parent.glob('*.py'):
            shutil.copy2(source, code/source.name)
        cfg = rs.config()
        if args.serial:
            cfg.enable_device(args.serial)
        cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
        cfg.enable_record_to_file(str((session.path / 'raw.bag').resolve()))
        profile = pipeline.start(cfg)
        started = True
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        if intr.model not in (rs.distortion.none, rs.distortion.brown_conrady) and any(intr.coeffs):
            raise ValueError('Unsupported distortion for AprilTag PnP')
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        save_json(session.path / 'camera.json', {'serial': profile.get_device().get_info(rs.camera_info.serial_number),
            'capture_run_id': run_id, 'width': intr.width, 'height': intr.height,
            'fx': intr.fx, 'fy': intr.fy, 'ppx': intr.ppx, 'ppy': intr.ppy,
            'distortion': str(intr.model), 'coeffs': intr.coeffs, 'depth_scale_m': scale,
            'native_recording': 'raw.bag', 'alignment': 'depth to color for inference only'})
        server = QuestServer(session, args.host, args.port)
        align = rs.align(rs.stream.color)

        def capture():
            try:
                while not stop.is_set():
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                    mono, epoch = time.monotonic_ns(), time.time_ns()
                    color, depth = frames.get_color_frame(), frames.get_depth_frame()
                    if not color or not depth:
                        continue
                    fid = f'{run_id}:{color.get_frame_number()}'
                    meta = {'frame_id': fid, 'capture_run_id': run_id, 'camera_id': args.camera_id,
                        'acquisition_monotonic_ns': str(mono), 'acquisition_epoch_ns': str(epoch),
                        'color_frame_number': color.get_frame_number(), 'depth_frame_number': depth.get_frame_number(),
                        'color_device_ms': color.get_timestamp(), 'depth_device_ms': depth.get_timestamp(),
                        'color_clock': str(color.get_frame_timestamp_domain()),
                        'depth_clock': str(depth.get_frame_timestamp_domain()), 'raw_locator': 'raw.bag',
                        'recorded': 'sdk_recording_requested_requires_playback_validation'}
                    session.emit('frames', meta)
                    with condition:
                        if slot[0] is not None:
                            session.emit('events', {'event': 'inference_frame_replaced', 'frame_id': slot[0][1]['frame_id']})
                        slot[0] = (frames, meta)
                        condition.notify()
            except Exception as e:
                errors.append(e)
                stop.set()
                with condition:
                    condition.notify_all()

        capture_thread = threading.Thread(target=capture, daemon=True)
        capture_thread.start()
        deadline = time.monotonic() + args.seconds if args.seconds else float('inf')
        sequence = 0
        last_status = 0
        print(f'Session {session.id}: {session.path}\nQuest ws://<PC>:{server.port}', flush=True)
        while time.monotonic() < deadline and not stop.is_set():
            if session.error:
                raise RuntimeError(session.error)
            with condition:
                condition.wait_for(lambda: slot[0] is not None or stop.is_set(), timeout=2)
                item, slot[0] = slot[0], None
            if item is None:
                continue
            frames, meta = item
            aligned = align.process(frames)
            image = np.asanyarray(aligned.get_color_frame().get_data()).copy()
            depth = np.asanyarray(aligned.get_depth_frame().get_data()).astype(np.float32) * scale
            result = infer.process(image, depth, intr)
            result.update(frame_id=meta['frame_id'], capture_run_id=run_id,
                          acquisition_monotonic_ns=meta['acquisition_monotonic_ns'])
            session.emit('skeleton', result)
            sequence += 1
            wire = packet(result, int(meta['acquisition_epoch_ns'])//1_000_000,
                          sequence, meta['frame_id'], session.id)
            # Preserve result even if too old for live control.
            if time.monotonic_ns()-int(meta['acquisition_monotonic_ns']) <= 350_000_000:
                server.publish(wire)
            else:
                session.emit('events', {'event': 'stale_downlink', 'frame_id': meta['frame_id']})
            if time.monotonic()-last_status > 1:
                print(f'inference={sequence} tags={len(result["tags"])} joints={len(result["joints"])} '
                      f'written={session.counts} queue={session.queue.qsize()}', flush=True)
                last_status = time.monotonic()
        if errors:
            raise errors[0]
    except KeyboardInterrupt:
        pass
    except Exception as e:
        failure = repr(e)
        raise
    finally:
        stop.set()
        if capture_thread:
            capture_thread.join(timeout=3)
        if server:
            server.close()
        if started:
            pipeline.stop()
        session.close(failure or (repr(errors[0]) if errors else None))

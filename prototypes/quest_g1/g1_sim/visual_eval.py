"""Closed-loop evaluation of a visual_training checkpoint in the Quest bimanual scene.

π_high sees the training camera (current physical ego camera, previous 30 Hz gaze,
640x360 without shadows -> 224x224) and the two previous 39-d commands. Its body35
output is the whole-body reference the Quest runtime would send after catch-up
(QuestSim, HOMIE v2); grips drive the hands; gaze only turns the camera. A chunk
is predicted with deterministic DDIM and its first ``execute`` actions are applied
at 30 Hz before replanning. Success is the native collection criterion.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

FPS = 30


class Camera:
    def __init__(self, model):
        import mujoco
        self.mj = mujoco; self.model = model
        model.light_castshadow[:] = 0
        self.renderer = mujoco.Renderer(model, height=360, width=640)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.camera = model.camera('ego').id

    def __call__(self, data, yaw, pitch):
        from PIL import Image
        # After mj_step body poses precede integration; refresh them from qpos only.
        self.mj.mj_kinematics(self.model, data); self.mj.mj_camlight(self.model, data)
        self.renderer.update_scene(data, camera=self.camera)
        cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
        rotation = data.cam_xmat[self.camera].reshape(3, 3) @ np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]) @ np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]])
        for cam in self.renderer.scene.camera:
            cam.pos[:] = data.cam_xpos[self.camera]; cam.forward[:] = -rotation[:, 2]; cam.up[:] = rotation[:, 1]
        self.renderer.scene.flags[self.mj.mjtRndFlag.mjRND_SHADOW] = 0
        return np.asarray(Image.fromarray(self.renderer.render()).resize((224, 224), Image.Resampling.BILINEAR))

    def close(self): self.renderer.close()


class PiHigh:
    def __init__(self, checkpoint, bounds, device='cpu', ddim_steps=20, seed=0):
        import torch
        from .visual_training import VisualDenoiser
        from .diffusion import cosine_alpha_bar
        self.torch = torch; self.device = device
        c = torch.load(checkpoint, map_location='cpu', weights_only=False)
        meta = c['metadata']
        self.model = VisualDenoiser(meta['width'], meta.get('arch', 'film'), meta.get('pretrained', False), meta.get('crop', 1.))
        self.model.load_state_dict(c['model'])
        self.model.to(device).eval()
        self.mean = c['action_mean'].numpy().astype(np.float32); self.std = c['action_std'].numpy().astype(np.float32)
        lo, hi = bounds
        self.lo = torch.tensor((lo-self.mean)/self.std, device=device); self.hi = torch.tensor((hi-self.mean)/self.std, device=device)
        self.alpha = cosine_alpha_bar(100).to(device)
        self.times = np.unique(np.linspace(len(self.alpha)-1, 0, ddim_steps).round().astype(int))[::-1]
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.horizon = c['metadata'].get('prediction_horizon', 64)

    def __call__(self, rgb, command):
        """rgb (2,224,224,3) uint8 [t-1,t]; command (2,39) previous commands. Returns (horizon,39)."""
        torch = self.torch
        with torch.no_grad():
            image = torch.tensor(rgb, device=self.device).permute(0, 3, 1, 2).float().div(255).unsqueeze(0)
            cmd = torch.tensor((command-self.mean)/self.std, device=self.device).unsqueeze(0)
            features = self.model.encode(image.reshape(2, 3, 224, 224)).reshape(1, 2, -1)
            condition = torch.cat((features, cmd), dim=-1)
            x = torch.randn((1, self.horizon, len(self.mean)), generator=self.generator, device=self.device)
            for i, t in enumerate(self.times):
                a = self.alpha[t]
                eps = self.model.denoiser(x, torch.full((1,), int(t), device=self.device), condition)
                x0 = ((x-(1-a).sqrt()*eps)/a.sqrt()).clamp(self.lo, self.hi)
                if i+1 == len(self.times): x = x0; break
                eps = (x-a.sqrt()*x0)/(1-a).sqrt(); n = self.alpha[self.times[i+1]]
                x = n.sqrt()*x0+(1-n).sqrt()*eps
        return x[0].cpu().numpy()*self.std+self.mean


def run_trial(sim, camera, policy, start, execute=8, seconds=40., record=False):
    """start: dict(seed) for a fresh layout, or dict(state=...) for a recorded start."""
    if 'state' in start:
        s = start['state']; sim.reset(int(start['seed'])); sim.load_state(*s)
    else:
        sim.reset(int(start['seed']))
        for _ in range(100): sim.cycle(None, 0, 0)  # 1 s standing before the operator takes over
    gaze = np.asarray(start.get('gaze', (0.0056, -0.823)), float)
    t0 = sim.data.time; frames, actions, video = [], [], []; chunk, base = None, 0
    status, tick, wall = 'timeout', 0, time.time()
    try:
        while sim.data.time-t0 < seconds:
            previous_gaze = actions[-1][37:39] if actions else gaze
            frames.append(camera(sim.data, *previous_gaze)); frames = frames[-2:]
            if record: video.append(frames[-1])
            if chunk is None or tick-base >= execute:
                rgb = np.stack([frames[0], frames[-1]])
                prev = lambda k: actions[k-1] if k >= 1 else np.zeros(39, np.float32)
                chunk = policy(rgb, np.stack([prev(max(tick-1, 0)), prev(tick)])); base = tick
            action = chunk[tick-base].astype(np.float32); actions.append(action)
            grip = np.clip(action[35:37], 0, 1)
            while sim.data.time-t0 < (tick+1)/FPS-1e-9:
                sim.cycle(action[:35], grip[0], grip[1])
                if sim.outcome: break
            tick += 1
            if sim.outcome: break
    except RuntimeError as exc:
        status = 'fell: ' + str(exc)
    outcome = sim.outcome
    if outcome == 1: status = 'success'
    elif outcome == 2: status = 'fell'
    u = sim.data.userdata
    result = dict(seed=int(start['seed']), source=start.get('episode'), status=status, outcome=outcome,
                  seconds=round(float(sim.data.time-t0), 2), lift_left=bool(u[4]), lift_right=bool(u[5]),
                  root_xy=[round(float(v), 3) for v in sim.data.qpos[:2]], wall=round(time.time()-wall, 1))
    return result, (np.stack(video) if record else None)


def recorded_starts(dataset, raw):
    """Validation episodes: recorded seed and physical state at the first demonstration frame."""
    import h5py
    from .visual_dataset import inspect_episode, columns, table
    starts = []
    with h5py.File(dataset, 'r') as h:
        names = [x.decode() for x in h['mask/valid'][:]]
        episodes = [h[f'data/{n}'].attrs['episode'] for n in names]
    for name in episodes:
        e = inspect_episode(Path(raw)/name); r = int(e['start']); f = e['frames']
        import mujoco
        m = mujoco.MjModel.from_binary_path(str(Path(raw)/name/'learning-model.mjb'))
        sc = table(Path(raw)/name/'scene_state.csv')
        state = (columns(f, 'qpos', m.nq)[r], columns(f, 'qvel', m.nv)[r], columns(sc, 'userdata', 8)[r],
                 columns(f, 'ctrl', m.nu)[r], f['step'][r])
        starts.append(dict(seed=int(sc['userdata_0'][0]), episode=name, state=state,
                           gaze=(float(e['camera']['gaze_yaw'][r]), float(e['camera']['gaze_pitch'][r]))))
    return starts


def action_bounds(dataset):
    import h5py
    with h5py.File(dataset, 'r') as h:
        a = np.concatenate([h[f'data/{n.decode()}/actions'][:] for n in h['mask/train'][:]])
    return a.min(0), a.max(0)


def worker(args):
    import torch
    from .quest_sim import QuestSim
    (index, starts, o) = args
    torch.set_num_threads(o['threads'])
    sim = QuestSim(o['model_episode']); camera = Camera(sim.model)
    policy = PiHigh(o['checkpoint'], o['bounds'], o['device'], o['ddim_steps'], seed=1000+index)
    out = []
    for k, start in enumerate(starts):
        record = o['videos'] and start.get('record', False)
        result, video = run_trial(sim, camera, policy, start, o['execute'], o['seconds'], record)
        if video is not None:
            path = Path(o['output']).with_name(f"{Path(o['output']).stem}-{result['source'] or result['seed']}.npz")
            np.savez_compressed(path, video=video)
        print(json.dumps(result), flush=True); out.append(result)
    camera.close()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint'); p.add_argument('dataset'); p.add_argument('raw'); p.add_argument('output')
    p.add_argument('--fresh', type=int, default=50, help='new layouts (seeds 1..N)')
    p.add_argument('--execute', type=int, default=8); p.add_argument('--ddim-steps', type=int, default=20)
    p.add_argument('--seconds', type=float, default=40.); p.add_argument('--workers', type=int, default=8)
    p.add_argument('--threads', type=int, default=2); p.add_argument('--device', default='cpu')
    p.add_argument('--videos', type=int, default=3, help='record this many fresh and valid trials')
    a = p.parse_args()
    import multiprocessing as mp
    output = Path(a.output)
    if output.exists(): raise FileExistsError(output)
    starts = recorded_starts(a.dataset, a.raw) + [dict(seed=s) for s in range(1, a.fresh+1)]
    for k, s in enumerate(starts):
        s['record'] = k < a.videos or (len(starts)-a.fresh <= k < len(starts)-a.fresh+a.videos)
    model_episode = str(Path(a.raw)/starts[0]['episode']) if 'episode' in starts[0] else str(next(Path(a.raw).glob('episode-*')))
    o = dict(checkpoint=a.checkpoint, bounds=action_bounds(a.dataset), device=a.device, ddim_steps=a.ddim_steps,
             execute=a.execute, seconds=a.seconds, threads=a.threads, videos=a.videos > 0, output=str(output),
             model_episode=model_episode)
    jobs = [(i, starts[i::a.workers], o) for i in range(a.workers)]
    t = time.time()
    with mp.get_context('spawn').Pool(a.workers) as pool:
        results = [r for chunk in pool.map(worker, jobs) for r in chunk]
    def summary(rows):
        n = len(rows)
        return dict(trials=n, success=sum(r['status'] == 'success' for r in rows),
                    fell=sum(r['status'].startswith('fell') for r in rows),
                    lift_left=sum(r['lift_left'] for r in rows), lift_right=sum(r['lift_right'] for r in rows),
                    both_lifted=sum(r['lift_left'] and r['lift_right'] for r in rows),
                    median_success_seconds=float(np.median([r['seconds'] for r in rows if r['status'] == 'success'])) if any(r['status'] == 'success' for r in rows) else None)
    report = dict(schema='furry-visual-closed-loop-v1', checkpoint=a.checkpoint, dataset=a.dataset,
                  settings=dict(execute=a.execute, ddim_steps=a.ddim_steps, seconds=a.seconds, fps=FPS,
                                initial='validation: recorded first demo frame; fresh: reset(seed 1..N) + 1 s standing',
                                note='HOMIE history at recorded starts is filled with the current observation'),
                  validation=summary([r for r in results if r['source']]), fresh=summary([r for r in results if not r['source']]),
                  wall_seconds=round(time.time()-t, 1), trials=results)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ('validation', 'fresh', 'wall_seconds')}, indent=1))


if __name__ == '__main__':
    main()

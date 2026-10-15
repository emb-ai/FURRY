"""Check g1_sim.quest_sim against recorded Quest collection episodes (bimanual, cup).

1. Reset parity: QuestSim.reset(seed) reproduces the recorded step-0 state (object layout RNG,
   sampled cup goal marker).
2. Teacher-forced HOMIE parity: recorded states and references in, predicted joint torques
   compared with recorded ctrl one 10 ms frame later (ctrl uses the state 1 ms earlier, so
   small residuals are expected; history is warmed up for 50 policy ticks).
Open-loop replay of whole episodes is not a parity test: walking and grasping diverge.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1_sim.quest_sim import QuestSim
from g1_sim.visual_dataset import table, columns

DEFAULT = Path(__file__).resolve().parents[3]/'datasets/quest-bimanual/2026-10-15/raw'


def check(path):
    import mujoco
    f = table(path/'frames.csv'); a = table(path/'mimic.csv'); sc = table(path/'scene_state.csv')
    sim = QuestSim(path); m, d = sim.model, sim.data
    qpos = columns(f, 'qpos', m.nq); qvel = columns(f, 'qvel', m.nv); ctrl = columns(f, 'ctrl', m.nu)
    mimic = columns(a, 'mimic', 35)
    sim.reset(int(sc['userdata_0'][0])); reset = float(np.abs(d.qpos-qpos[0]).max())
    if m.nmocap: reset = max(reset, float(np.abs(d.mocap_pos.ravel()-columns(sc, 'mocap_pos', 3*m.nmocap)[0]).max()))
    errors = []
    for r in range(1, len(f['step'])-1):
        if f['step'][r] % 20: continue
        d.qpos[:] = qpos[r]; d.qvel[:] = qvel[r]; mujoco.mj_forward(m, d)
        if f['apply_reference'][r] or mimic[r].any(): sim.set_reference(mimic[r])
        sim._step_decoupled()
        torque = sim.torque(qpos[r+1], qvel[r+1])
        errors.append(np.abs(torque-ctrl[r+1][sim.ai]).max())
    return reset, float(np.median(errors[50:])), float(np.percentile(errors[50:], 99))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('episodes', nargs='*', type=Path, help='default: first 3 bimanual episodes')
    a = p.parse_args()
    paths = a.episodes or sorted(DEFAULT.glob('episode-*'))[:3]
    for path in paths:
        reset, median, p99 = check(path)
        print(f'{path.name}: reset max error {reset:.2e}, torque residual median {median:.3f} Nm, p99 {p99:.3f} Nm')
        assert reset < 1e-9, 'reset/object layout differs from the recording'
        assert median < .5 and p99 < 3, 'HOMIE port differs from the native backend'
    print('QuestSim parity checks passed')


if __name__ == '__main__':
    main()

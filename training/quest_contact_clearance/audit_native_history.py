"""Instrument a COPY of native Simulation::Step and compare its 1432-vector.

Only history/tail assembly is compared here. Current proprio is copied from
native, so this test does not assert Isaac-to-MuJoCo proprio/dynamics parity.
Build/run with the existing native Python environment, compare with Torch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

DRIVER = r'''
#include "simulation.h"
#include <fstream>
#include <iomanip>
std::ofstream capture;
void CaptureActorObservation(const std::array<float,1432>& obs) {
    for (int i=0;i<1432;i++) capture << (i ? "," : "") << obs[i];
    capture << '\n';
}
int main(int argc,char** argv) {
    if(argc!=3)return 2;
    capture.open(argv[2]); capture << std::setprecision(9);
    Simulation s(argv[1],0);
    for (int tick=0;tick<14;tick++) {
        if(tick==7)s.Reset();
        std::array<float,35> cmd{};cmd[0]=.02*tick;cmd[2]=.78;
        cmd[5]=.03*tick;cmd[6]=-.1;cmd[9]=.3;cmd[12]=-.1;cmd[15]=.3;
        s.SetWholeBodyReference(cmd);
        for(int k=0;k<1000/s.PolicyHz();k++)s.Step(false);
    }
}
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--compare', action='store_true')
    a = p.parse_args()
    if a.compare:
        import numpy as np
        import torch
        from native_command_bank import QuestActorHistory
        obs = np.loadtxt(a.output/'native-observations.csv', delimiter=',', dtype=np.float32)
        assert obs.shape == (14, 1432) and np.isfinite(obs).all()
        history = QuestActorHistory(1, 'cpu'); maximum = 0.
        for i, row in enumerate(obs):
            if i == 7:
                history.reset(torch.tensor([0]))
            got = history.observe(torch.from_numpy(row[None, :127])).numpy()[0]
            maximum = max(maximum, float(np.abs(got-row).max()))
        result = {'ticks': len(obs), 'resets': 2, 'maximum_absolute_error': maximum,
                  'passed': maximum == 0., 'scope': 'actual native history/tail assembly only; no Isaac dynamics/proprio parity claim'}
        (a.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result)); assert result['passed']
        return
    import mujoco
    import onnxruntime
    a.output.mkdir(parents=True, exist_ok=True)
    native = a.source.resolve()/'android/native'
    source = (native/'simulation.cpp').read_text()
    prefix, boundary, step_source = source.partition('void Simulation::Step(')
    assert boundary
    anchor = 'std::array<int64_t,2> shape{1,1432};'
    assert step_source.count(anchor) == 1
    source = '#include <array>\nextern void CaptureActorObservation(const std::array<float,1432>&);\n' + prefix + boundary + step_source.replace(anchor, 'CaptureActorObservation(obs);\n        '+anchor)
    (a.output/'simulation-instrumented.cpp').write_text(source)
    (a.output/'driver.cpp').write_text(DRIVER)
    mj = Path(mujoco.__file__).parent; ort = Path(onnxruntime.__file__).parent/'capi'
    binary = a.output/'audit-native-history'
    argv = ['c++', '-O2', '-std=c++17', '-I'+str(native), '-I'+str(mj/'include'),
            '-I'+str(a.source.resolve()/'vendor/onnxruntime-android/headers'),
            str(a.output/'driver.cpp'), str(a.output/'simulation-instrumented.cpp'),
            str(next(mj.glob('libmujoco*.dylib'))), str(next(ort.glob('libonnxruntime*.dylib'))), '-o', str(binary)]
    subprocess.run(argv, check=True)
    subprocess.run([str(binary), str(a.source.resolve()/'android/assets'), str(a.output/'native-observations.csv')],
                   env=dict(os.environ, DYLD_LIBRARY_PATH=os.pathsep.join(map(str, (mj, ort)))), check=True)
    (a.output/'source.json').write_text(json.dumps({'simulation_cpp_sha256': hashlib.sha256((native/'simulation.cpp').read_bytes()).hexdigest(),
                                                 'argv': argv}, indent=2)+'\n')


if __name__ == '__main__':
    main()

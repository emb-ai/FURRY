"""User-authorized inverse compilation keeps the first actor's vocabulary small."""
import numpy as np

# Native face numbering. Clockwise seen from outside is -90 about its normal.
FACES = {'R':0, 'F':3, 'U':4}

def compile_sequence(commands):
    quarters, logical_ends = [], []
    for command in commands:
        if command not in tuple(FACES)+tuple(k+"'" for k in FACES):
            raise ValueError(f'Unsupported command: {command}')
        face=FACES[command[0]]
        quarters.extend([(face,-1)]*(3 if command.endswith("'") else 1))
        logical_ends.append(len(quarters))
    return quarters, logical_ends

def random_sequence(seed, length):
    rng=np.random.default_rng(seed)
    vocabulary=tuple(FACES)+tuple(k+"'" for k in FACES)
    return [str(x) for x in rng.choice(vocabulary,size=length)]

if __name__=='__main__':
    from native_env import ROOT, turn_target
    from scipy.spatial.transform import Rotation as R
    z=np.load(ROOT.parent/'rubik-feedback-20261015/verified-nine.npz')
    initial=z['userdata'][4049,8:64].reshape(8,7)
    for label,face in FACES.items():
        compiled,_=compile_sequence([label+"'"])
        current=initial.copy()
        for command in compiled:current=turn_target(current,*command)
        expected=turn_target(initial,face,1)
        assert np.max(abs(current[:,:3]-expected[:,:3]))<1e-12
        assert (R.from_quat(current[:,[4,5,6,3]])*R.from_quat(expected[:,[4,5,6,3]]).inv()).magnitude().max()<1e-12
    assert random_sequence(1,20)==random_sequence(1,20)
    print('All three inverse compilations and seeded random sequences verified.')

"""Native replay snapshots at episode boundaries; continuous wrist/Grip actions inside episodes."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from native_env import NativeRubikEnv, ROOT, lib, turn_target, CONTROL_DT

BOUNDS = np.array([40.5,59.32,78.17,93.02,117.01,131.90,148.16,164.62,180.98,197.29])
VALID_MACROS = (3, 7)
OFFSETS = (.6, 1., 1.5, 2., 3., 5., 8.)
STAGES = [
    dict(name='settle_grasp', offsets=(.6,1.), timeout=5., length=1, faces=(3,)),
    dict(name='close_grasp', offsets=(1.5,2.), timeout=10., length=1, faces=(3,)),
    dict(name='approach_grasp', offsets=(3.,5.), timeout=18., length=1, faces=(3,)),
    dict(name='release_withdraw_return', offsets=(8.,'return_start'), timeout=35., length=1, faces=(3,)),
    dict(name='complete_F', offsets=(), timeout=55., length=1, faces=(3,)),
    dict(name='random_FU_2', offsets=(), timeout=120., length=2, faces=(3,4)),
    dict(name='random_RFU_2', offsets=(), timeout=120., length=2, faces=(0,3,4)),
    dict(name='random_RFU_5', offsets=(), timeout=120., length=5, faces=(0,3,4)),
    dict(name='random_RFU_10', offsets=(), timeout=120., length=10, faces=(0,3,4)),
]

class CurriculumEnv(NativeRubikEnv):
    def __init__(self, stage=0, evaluation=False, seed=0):
        self.stage = int(stage)
        self.evaluation = evaluation
        self.bank = []
        self.current_start = None
        self.bank_ready = False
        super().__init__(seed=seed)
        self._build_bank()
        self.bank_ready = True
        self.reset(seed=seed)

    def _build_bank(self):
        path = ROOT.parent/'rubik-feedback-20261015/verified-nine.npz'
        z = dict(np.load(path))
        requests = {}
        # Return starts after a genuine layer turn, before the first opening.
        # These states must still learn release and withdrawal after reset.
        for macro,(a,b) in enumerate(zip(BOUNDS,BOUNDS[1:])):
            ai,bi = round(a*100),round(b*100)
            cmds = z['commands'][ai:bi]
            open_at = np.flatnonzero(cmds[:,-1] < .1)
            if not len(open_at): raise RuntimeError('No release in source macro')
            first_open = ai + int(open_at[0])
            for offset in (*OFFSETS,'return_start'):
                frame = first_open if offset=='return_start' else round((b-offset)*100)
                requests.setdefault(frame,[]).append((macro,offset))
        lib.restore(self.s.p,self.initial_snapshot)
        self.s.read()
        macro = 0
        max_error = 0.
        histories = {}
        for k in range(4050,round(BOUNDS[-1]*100)):
            while k >= round(BOUNDS[macro+1]*100):
                print(f'BANK replay macro {macro+1}/9 pid={__import__("os").getpid()}',flush=True)
                macro += 1
            if k == round(BOUNDS[macro]*100):
                logical = z['userdata'][k-1,8:64].reshape(8,7).copy()
                distance0 = np.linalg.norm(self.s.d.xpos[self.geometry.wr]-self.s.d.xpos[self.geometry.cube].mean(0),axis=1)
                ticks = np.zeros(2,dtype=int); released=False
            # frame=k denotes the state AFTER k recorded commands, before command k.
            if k in requests:
                for mi,offset in requests[k]:
                    assert mi==macro
                    error = float(np.max(abs(self.s.q-z['qpos'][k-1])))
                    max_error=max(max_error,error)
                    if error>1e-9: raise RuntimeError(f'Warm snapshot replay mismatch {error}')
                    self.geometry.update(self.s.q)
                    target=turn_target(logical,3,-1)
                    if self.geometry.metrics(target)['corner_angle']>np.deg2rad(3):
                        raise RuntimeError('Recovery snapshot does not preserve a completed face turn')
                    self.bank.append(dict(snapshot=lib.snapshot(self.s.p),macro=macro,offset=offset,frame=k,
                        logical=logical.copy(),goals=z['commands'][k-1,:14].reshape(2,7).copy(),
                        grips=z['commands'][k-1,-2:].copy(),released=bool(released),
                        release_ticks=ticks.copy(),start_distance=distance0.copy()))
            cc=z['commands'][k]
            if self.s.step(cc[:14].reshape(2,7),cc[-2:]):raise RuntimeError('Native failure building bank')
            masks=np.array([lib.rubik_contacts(self.s.p,h) for h in range(2)])
            if not np.any(masks) or lib.supports(self.s.p):raise RuntimeError('Source violates airborne constraint')
            ticks=np.where((masks==0)&(cc[-2:]<.1),ticks+1,0)
            c=self.s.d.xpos[self.geometry.cube].mean(0)
            dist=np.linalg.norm(self.s.d.xpos[self.geometry.wr]-c,axis=1)
            released |= bool(np.any((ticks>=20)&(dist>distance0+.02)))
        assert len(self.bank)==9*(len(OFFSETS)+1)
        self.bank_report=dict(states=len(self.bank),max_qpos_error=max_error,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            snapshot_restore_only_at_episode_reset=True,validation_macros=list(VALID_MACROS),
            independent_demonstrations=False)

    def set_stage(self, stage):
        # Applied only to the next reset; never overwrite an active episode.
        self.stage=int(stage)
        if self.stage not in range(len(STAGES)):raise ValueError(stage)

    def reset(self, *, seed=None, options=None):
        if not self.bank_ready:return super().reset(seed=seed,options=options)
        cfg=STAGES[self.stage]
        self.move_timeout=cfg['timeout']
        options=options or {}
        # Seed native Gym RNG before drawing the queue and snapshot.
        if seed is not None:self.np_random=np.random.default_rng(seed)
        if 'sequence' in options:
            sequence=[tuple(x) for x in options['sequence']]
            self.logical_end_indices=list(range(1,len(sequence)+1))
        elif self.stage<5:
            sequence=[(3,-1)];self.logical_end_indices=[1]
        else:
            sequence=[];self.logical_end_indices=[];last=None
            for _ in range(cfg['length']):
                face=int(self.np_random.choice([f for f in cfg['faces'] if f!=last]))
                # User permits inverse turns as three positive quarters.
                inverse=bool(self.np_random.integers(2))
                sequence.extend([(face,-1)]*(3 if inverse else 1))
                self.logical_end_indices.append(len(sequence));last=face
        obs,info=super().reset(options={'sequence':sequence})
        self.current_start=None
        if cfg['offsets'] and not options.get('cold',False):
            candidates=[e for e in self.bank if (e['macro'] in VALID_MACROS)==self.evaluation and e['offset'] in cfg['offsets']]
            if 'bank_index' in options:
                entry=self.bank[int(options['bank_index'])]
                if not any(entry is e for e in candidates):raise ValueError('Snapshot outside this split/stage')
            else:entry=candidates[int(self.np_random.integers(len(candidates)))];
            lib.restore(self.s.p,entry['snapshot']);self.s.read();self.geometry.update(self.s.q)
            self.goals=entry['goals'].copy();self.grips=entry['grips'].copy();self.logical=entry['logical'].copy()
            self._next_goal()
            self.released=entry['released'];self.release_ticks=entry['release_ticks'].copy()
            self.start_distance=entry['start_distance'].copy()
            self.current_start=dict(macro=entry['macro'],offset=entry['offset'],frame=entry['frame'])
            obs=self.observe()
        self.episode_stage=self.stage
        self.episode_steps=0
        info.update(stage=cfg['name'],warm_start=self.current_start,logical_end_indices=self.logical_end_indices)
        return obs,info

    @staticmethod
    def _potential(m):
        # Dense recovery signal without relaxing the physical success thresholds.
        return -(2*m['corner_position']/.04 + 2*m['corner_angle']/(np.pi/2)
                 + .5*m['center']/.01 + .5*max(m['hand_position'])/.01
                 + .2*max(m['hand_angle'])/.1 + .3*m['grid_angle']/.1)

    def step(self, action):
        obs,reward,done,truncated,info=super().step(action)
        self.episode_steps+=1
        info.update(stage=self.episode_stage,warm_start=self.current_start,
            completed_logical_moves=sum(self.completed>=i for i in self.logical_end_indices),
            requested_logical_moves=len(self.logical_end_indices),
            elapsed_seconds=self.episode_steps*CONTROL_DT)
        return obs,reward,done,truncated,info

    def close(self):
        for e in self.bank:lib.free_snapshot(e['snapshot'])
        self.bank=[]
        super().close()

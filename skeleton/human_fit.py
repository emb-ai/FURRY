"""Experimental, stateless human leg IK. Does not command the robot.

Lengths belong to the operator, in metres, and stay fixed during evaluation.
No hidden-joint reconstruction without a current explicitly labelled prior.
Camera landmarks are surface estimates, not anatomical ground truth.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

NAMES = ('lhip', 'rhip', 'lknee', 'rknee', 'lankle', 'rankle')
EDGES = ((0, 1), (0, 2), (1, 3), (2, 4), (3, 5))


def observed(joint):
    return (joint.get('src') == 'window' and joint.get('conf', 0) >= .55
            and np.asarray(joint.get('p', [])).shape == (3,)
            and np.isfinite(joint['p']).all())


@dataclass
class HumanLegFit:
    lengths: np.ndarray
    max_adjustment: float = .15

    def __post_init__(self):
        self.lengths = np.asarray(self.lengths, dtype=float)
        if (self.lengths.shape != (5,) or not np.isfinite(self.lengths).all()
                or np.any(self.lengths <= .05) or np.any(self.lengths >= 1.)):
            raise ValueError('Expected hip width and four positive leg lengths in metres')

    @classmethod
    def calibrate(cls, packets, minimum=30):
        """Robust provisional lengths from a separate calibration interval.

        Reject insufficient/unstable calibration instead of guessing height ratios.
        A tape-measured profile can replace these camera-derived lengths.
        """
        samples = []
        for p in packets:
            if not all(observed(p['joints'][n]) for n in NAMES):
                continue
            x = np.array([p['joints'][n]['p'] for n in NAMES])
            lengths = np.array([np.linalg.norm(x[a]-x[b]) for a,b in EDGES])
            if .08 < lengths[0] < .5 and np.all((lengths[1:] > .2) & (lengths[1:] < .65)):
                samples.append(lengths)
        if len(samples) < minimum:
            raise ValueError('Not enough complete measured calibration frames')
        med = np.median(samples, axis=0)
        mad = np.median(np.abs(samples-med), axis=0)
        if np.any(mad / med > .08):
            raise ValueError('Unstable length calibration; use a new calibration interval')
        return cls(med)

    def fit(self, packet, prior=None):
        """Return separate fitted points/provenance; leave input packet untouched.

        Optional prior is current-frame camera-metric geometry, never last pose.
        A prior is necessary when a point is missing. It remains 'predicted'.
        Knee flexion <=160 degrees; bend plane/twist cannot be validated from
        three positions alone. This is not a complete anatomical joint model.
        """
        joints = packet['joints']
        measured = np.array([observed(joints[n]) for n in NAMES])
        x, weights = [], []
        for n, good in zip(NAMES, measured):
            if good:
                x.append(joints[n]['p']); weights.append(joints[n]['conf'])
            elif prior is not None and n in prior and np.asarray(prior[n]).shape == (3,) and np.isfinite(prior[n]).all():
                x.append(prior[n]); weights.append(.05)
            else:
                return {'accepted': False, 'reason': 'missing_current_observation_or_prior'}
        if measured.sum() < 3:
            return {'accepted': False, 'reason': 'insufficient_metric_anchors'}
        target = np.array(x, dtype=float); weights = np.array(weights)

        def objective(flat):
            d = flat.reshape(6,3)-target
            # Smooth robust loss, scale 3 cm. No temporal lag or future frames.
            return np.sum(weights*.03**2*(np.sqrt(1+np.sum((d/.03)**2,axis=1))-1))

        def gradient(flat):
            d = flat.reshape(6,3)-target
            return (weights[:,None]*d/np.sqrt(1+np.sum((d/.03)**2,axis=1))[:,None]).ravel()

        def bones(flat):
            p = flat.reshape(6,3)
            return np.array([np.dot(p[a]-p[b],p[a]-p[b])-l*l
                             for (a,b),l in zip(EDGES,self.lengths)])

        def bone_jac(flat):
            p=flat.reshape(6,3); j=np.zeros((5,6,3))
            for k,(a,b) in enumerate(EDGES):
                j[k,a]=2*(p[a]-p[b]); j[k,b]=-j[k,a]
            return j.reshape(5,18)

        def knees(flat):
            p=flat.reshape(6,3)
            return np.array([np.cos(np.deg2rad(20))*self.lengths[1+s]*self.lengths[3+s]
                             -np.dot(p[s]-p[2+s],p[4+s]-p[2+s]) for s in range(2)])

        solved=minimize(objective,target.ravel(),jac=gradient,method='SLSQP',
                        constraints=[{'type':'eq','fun':bones,'jac':bone_jac},
                                     {'type':'ineq','fun':knees}],
                        options={'maxiter':60,'ftol':1e-9})
        p=solved.x.reshape(6,3)
        residual=np.linalg.norm(p-target,axis=1)
        valid=(solved.success and np.isfinite(p).all()
               and np.max(np.abs(bones(solved.x)))<1e-5
               and np.min(knees(solved.x))>=-1e-6
               and np.max(residual[measured])<=self.max_adjustment)
        return {'accepted':bool(valid), 'reason':'ok' if valid else 'inconsistent_geometry_or_solver_failure',
                'points':dict(zip(NAMES,p.tolist())) if valid else {},
                'sources':dict(zip(NAMES,['fitted' if g else 'predicted' for g in measured])) if valid else {},
                'max_adjustment_m':float(np.max(residual[measured])),
                'iterations':int(solved.nit)}

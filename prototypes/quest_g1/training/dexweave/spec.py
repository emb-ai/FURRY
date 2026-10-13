"""Versioned body-only contract, anatomical joint order, paper settings."""
from dataclasses import asdict, dataclass

JOINTS = tuple([f'{s}_{p}_joint' for s in ('left','right') for p in
    ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')] +
    ['waist_yaw_joint','waist_roll_joint','waist_pitch_joint'] +
    [f'{s}_{p}_joint' for s in ('left','right') for p in
     ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')])
LINKS = ('pelvis','left_hip_roll_link','left_knee_link','left_ankle_roll_link',
         'right_hip_roll_link','right_knee_link','right_ankle_roll_link','torso_link',
         'left_shoulder_roll_link','left_elbow_link','left_wrist_yaw_link',
         'right_shoulder_roll_link','right_elbow_link','right_wrist_yaw_link')
GROUPS = {'left_leg':range(0,6),'right_leg':range(6,12),'waist':range(12,15),
          'left_arm':range(15,22),'right_arm':range(22,29)}
# Actor order: reference q, reference dq, anchor xyz, anchor rot6(row-major
# first TWO COLUMNS), base angular velocity, current q-default, current dq,
# previous APPLIED normalized action. Quaternion convention everywhere: wxyz.
ACTOR_SLICES = {'reference_q':(0,29),'reference_dq':(29,58),'anchor_pos':(58,61),
                'anchor_rot6':(61,67),'base_omega':(67,70),'q_rel':(70,99),
                'dq':(99,128),'last_action':(128,157)}
SCHEMA = 'furry-dexweave-body-v1'
@dataclass(frozen=True)
class PaperSettings:
    actor_dim:int=157
    critic_dim:int=286
    action_dim:int=29
    control_hz:int=50
    physics_hz:int=200
    num_envs:int=4096
    rollout_steps:int=24
    learning_epochs:int=5
    minibatches:int=4
    learning_rate:float=1e-3
    desired_kl:float=.01
    clip:float=.2
    gamma:float=.99
    gae_lambda:float=.95
    entropy:float=.005
    max_grad_norm:float=1.
    initial_std:float=1.
    iterations:int=8000
    max_episode_steps:int=500

# Exact scalars published for body-only, Table8 + following paragraph.
REWARDS = {'anchor_pos':(1.,.20),'anchor_ori':(.75,.30),
           'body_pos':(1.,.30),'body_ori':(1.,.40),
           'body_lin_vel':(1.,1.),'body_ang_vel':(1.,3.14),'joint_pos':(1.,.40),
           'action_first_difference':(-.10,None),'waist_second_difference':(-.05,None),
           'joint_limits':(-10.,None),'undesired_contacts':(-.10,None)}

def contract():
    return {'schema':SCHEMA,'paper':'https://arxiv.org/html/2609.34724v1',
            'source_release_available':False,'implementation':'independent paper reconstruction',
            'leg_source':'meta_virtual_legs','joints':JOINTS,'links':LINKS,
            'actor_slices':ACTOR_SLICES,'settings':asdict(PaperSettings()),'rewards':REWARDS,
            'unpublished_details':[
                'Body-only head context for waist/arms not fully specified: base+waist and waist+same arm selected.',
                'Transformer pre/post norm not specified: post-norm selected.',
                'Continuous foot contact score formulas not fully specified: explicit Quest ankle-based adaptation.',
                'Exact author G1 body PD/model unavailable: pinned BeyondMimic baseline to be verified.']}

"""Causal desired swing independent of actual foot motion/contact.

Recorded references use uncorrected world sole-centroid displacement. Held
operator commands use an explicit 1 Hz alternating schedule, with 60% stance
and a quintic rise/fall to 65 mm. These are experimental parameters, inspired
by phase-conditioned locomotion (HUGWBC), not a reproduction of that controller.
"""
import math
import torch


def intent_cost(height, weight, target):
    # Same physical units/exponential denominator as the old cost. Only the
    # source of weighting/target changes; no actual speed or contact argument.
    return -torch.expm1(-(weight*(target-height).clamp(min=0).square()).sum(-1)/.05)


def scheduled_swing(phase):
    foot_phase=(phase[:,None]+phase.new_tensor([0.,.5]))%1.
    u=((foot_phase-.6)/.4).clamp(0,1)
    rise=(2*torch.minimum(u,1-u)).clamp(0,1)
    profile=rise.pow(3)*(10-15*rise+6*rise.square())
    return (foot_phase>.6).to(phase.dtype),.065*profile


class SwingIntent:
    def __init__(self,n,device,dt=.02):
        self.dt=dt
        self.previous=torch.zeros(n,2,3,device=device)
        self.valid=torch.zeros(n,dtype=torch.bool,device=device)
        self.phase=torch.zeros(n,device=device)
        self.active=torch.zeros(n,dtype=torch.bool,device=device)
        self.weight=torch.zeros(n,2,device=device)
        self.target=torch.zeros_like(self.weight)

    def reset(self,ids):
        self.valid[ids]=False;self.phase[ids]=0;self.active[ids]=False
        self.weight[ids]=0;self.target[ids]=0

    def update(self,raw_centroid,held,command_xy,command_yaw):
        # Invalid first differences never create an artificial requested step.
        speed=torch.linalg.vector_norm((raw_centroid-self.previous)[...,:2]/self.dt,dim=-1)
        source_weight=torch.tanh(2*speed)*self.valid[:,None]
        self.previous.copy_(raw_centroid);self.valid[:]=True
        demand=(torch.linalg.vector_norm(command_xy,dim=-1)>.05)|(command_yaw.abs()>.10)
        # Finish a started cycle when the command stops, so a swing's height
        # target does not suddenly drop to zero midway through that swing.
        start=held&demand&~self.active
        self.active|=start;self.phase=torch.where(start,torch.zeros_like(self.phase),self.phase)
        advanced=self.phase+self.dt
        finish=self.active&(advanced>=1.)&~demand
        self.phase=torch.where(self.active,advanced%1.,torch.zeros_like(self.phase))
        self.active&=held&~finish
        swing,target=scheduled_swing(self.phase)
        swing*=self.active[:,None];target*=self.active[:,None]
        self.weight=torch.where(held[:,None],swing,source_weight)
        self.target=torch.where(held[:,None],target,torch.full_like(target,.065))

    def features(self):
        clock=torch.stack((torch.sin(2*math.pi*self.phase),torch.cos(2*math.pi*self.phase)),-1)
        clock*=self.active[:,None]
        return torch.cat((self.weight,self.target/.065,clock),-1)

    def cost(self,height):
        return intent_cost(height,self.weight,self.target)


def initialize_with_extra_inputs(model,state):
    """Preserve old policy output exactly by padding only input columns with 0.

    Optimizer intentionally starts fresh for a new objective/input contract.
    Ordinary subsequent block resumes retain the complete optimizer as before.
    """
    wanted=model.state_dict();result={}
    allowed={'actor.actor_backbone.0.weight','critic.0.weight'}
    for k,v in wanted.items():
        old=state[k]
        if old.shape==v.shape:result[k]=old
        elif k in allowed and old.ndim==2 and old.shape[0]==v.shape[0] and old.shape[1]+6==v.shape[1]:
            result[k]=torch.cat((old,torch.zeros(old.shape[0],6,device=old.device,dtype=old.dtype)),1)
        else:raise ValueError('Unsupported initialization shape: '+k)
    if set(state)!=set(wanted):raise ValueError('Initialization key mismatch')
    model.load_state_dict(result)

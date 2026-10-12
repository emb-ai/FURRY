"""Phase-free low-moving-foot cost prototype; not a trained policy.

Uses the published Unitree G1 velocity-weighted clearance reward family, with
explicit adaptations: whole physical sole minimum height, a one-sided deficit,
same-point sole-centroid world XY speed, and a bounded penalty instead of a
positive symmetric target-height reward. No desired phase or contact gate.

All distances are metres, velocities m/s, denominator m^2, speed gain s/m.
The default experiment is h_min=.065 = official ankle-link target .1 minus
the horizontal sole offset .035 in the exact trainer sphere URDF. For a pitched
foot this minimum-over-spheres target is more conservative than a link target.
This is a target parameter, not a pose clamp or retargeting edit.
"""
from dataclasses import asdict, dataclass
import math
import torch


@dataclass(frozen=True)
class ClearanceConfig:
    min_sole_clearance_m: float = .065
    speed_gain_s_per_m: float = 2.
    exponential_denominator_m2: float = .05
    reward_coefficient: float = -1.

    def __post_init__(self):
        for name in ('min_sole_clearance_m', 'speed_gain_s_per_m', 'exponential_denominator_m2'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<=0:
                raise ValueError(name+' must be finite and positive')
        if not math.isfinite(self.reward_coefficient) or self.reward_coefficient>=0:
            raise ValueError('reward_coefficient must be finite and negative')


def velocity_clearance_cost(clearance_m, velocity_xy_m_s, config=ClearanceConfig()):
    """Return batch cost [0,1] and unnormalized per-foot exponent numerator.

    Shape: heights [..., feet], XY velocity [..., feet, 2]. Caller selects actual
    speed in the planned experiment; reference speed is a separate CPU-only
    counterexample. No reference height, phase label, contact force or action
    enters this formula. Invalid simulated state remains invalid (not zeroed).
    """
    if clearance_m.shape!=velocity_xy_m_s.shape[:-1] or velocity_xy_m_s.shape[-1]!=2:
        raise ValueError('Expected matching [...,feet] heights and [...,feet,2] velocities')
    speed=torch.linalg.vector_norm(velocity_xy_m_s,dim=-1)
    deficit=(config.min_sole_clearance_m-clearance_m).clamp(min=0)
    terms=deficit.square()*torch.tanh(config.speed_gain_s_per_m*speed)
    return -torch.expm1(-terms.sum(-1)/config.exponential_denominator_m2),terms


def sole_geometry_cost(points,previous_points,radii,dt,config=ClearanceConfig()):
    """Use physical sphere bottoms and same-centroid FD world XY velocity.

    A 20ms difference in training is an average over the policy interval, not an
    instantaneous Isaac velocity and not a 1kHz solver measurement. The centroid
    choice avoids mixing Isaac COM linear velocity and link-origin positions.
    No contact-dependent selection of points affects the clearance weight.
    """
    if points.shape!=previous_points.shape or points.shape[-1]!=3:
        raise ValueError('Expected matching sphere-centre XYZ arrays')
    clearance=(points[...,2]-radii).min(-1).values
    velocity=(points.mean(-2)-previous_points.mean(-2))/dt
    cost,terms=velocity_clearance_cost(clearance,velocity[...,:2],config)
    return cost,terms,clearance,velocity


def contact_point_slip(points,previous_points,radii,angular_velocity,force_z,dt,valid,body_weight):
    """Unchanged v1 all-contact slip approximation, independent of ref phase.

    Requires the enforced plane/single-actor training scene. Net foot force is a
    floor-force proxy gated by physical sole proximity, not exact contact-pair
    force. Only this slip term reads contact forces; clearance does not.
    """
    arm=torch.zeros_like(points);arm[...,2]=-radii
    velocity=(points-previous_points)/dt+torch.cross(angular_velocity[...,None,:].expand_as(points),arm,dim=-1)
    bottom=points[...,2]-radii;clearance=bottom.min(-1).values
    near=(bottom<=.008)&(bottom<=clearance[...,None]+.008)
    speed2=velocity[...,:2].square().sum(-1)
    speed=torch.sqrt((speed2*near).sum(-1)/near.sum(-1).clamp(min=1))
    floor=(clearance<=.008)&(force_z>5)&valid[:,None]
    load=(force_z.clamp(min=0)/(body_weight/2)).clamp(max=2)
    return (torch.sqrt(speed.clamp(min=0))*load*floor).sum(-1)


def provenance(config=ClearanceConfig()):
    return dict(config=asdict(config),formula='1-exp(-sum(max(h_min-h_sole,0)^2*tanh(2*||v_sole_centroid_world_xy||))/.05)',
        training_reward='reward_coefficient * dt * cost; coefficient=-1, dt=.02 exactly once',
        source_target_conversion='.100m ankle link origin - .035m horizontal sphere sole offset = .065m',
        source_denominator='.05 directly, not .05 squared',
        actor_inputs=1432,critic_inputs_with_existing_window=1743,
        phase_classifier_gating=False,reference_height_used=False,contact_gating=False,
        limitations=['Actual-speed weight permits frozen-foot escape; unchanged tracking and directed progress must reject it',
            'Intentional glides and off-centre pivots can incur a cost; this is not a swing-intent classifier',
            'Whole-sole minimum target is stricter for pitched feet than the source ankle-link height',
            'One-sided bounded penalty changes constant reward and high-foot shaping from the source positive symmetric reward',
            'No trained-policy benefit follows from these algebraic and recorded-state checks'])


class ClearanceState:
    """Same causal actual-state diagnostics for both experiment conditions."""
    def __init__(self,env,urdf,config=ClearanceConfig()):
        from sole_kinematics import SoleKinematics
        self.env=env;self.config=config
        self.kinematics=SoleKinematics(urdf,env.gym.get_actor_dof_names(env.envs[0],env.actor_handles[0]),env.device)
        self.previous_points=torch.zeros((env.num_envs,2,self.kinematics.points.shape[1],3),device=env.device)
        self.valid=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
        self.terms=[torch.zeros(env.num_envs,device=env.device) for _ in range(2)]
        self.body_weight=torch.tensor([sum(p.mass for p in env.gym.get_actor_rigid_body_properties(e,a))*9.81
                                      for e,a in zip(env.envs,env.actor_handles)],device=env.device)[:,None]
        self.metric_sum=torch.zeros(5,device=env.device)
        self.samples=torch.zeros((),device=env.device)

    def reset(self,ids):
        self.valid[ids]=False

    def advance(self):
        e=self.env;k=self.kinematics;rb=e.rigid_body_states[:,e.feet_indices]
        points=k.world_soles(rb[...,:3],rb[...,3:7])
        cost,numerator,height,velocity=sole_geometry_cost(points,self.previous_points,k.radii,e.dt,self.config)
        slip=contact_point_slip(points,self.previous_points,k.radii,rb[...,10:13],e.contact_forces[:,e.feet_indices,2],e.dt,self.valid,self.body_weight)
        self.terms=[slip,torch.where(self.valid,cost,torch.zeros_like(cost))]
        speed=torch.linalg.vector_norm(velocity[...,:2],dim=-1)
        mask=self.valid[:,None].expand_as(height)
        deficit=(self.config.min_sole_clearance_m-height).clamp(min=0)
        self.metric_sum+=torch.stack(((numerator*mask).sum(),(speed*mask).sum(),
            (deficit*mask).sum(),(height*mask).sum(),self.terms[1].sum()))
        self.samples+=mask.sum()
        self.previous_points=points.clone();self.valid[:]=True

    def metrics(self):
        n=self.samples.clamp(min=1)
        values=(self.metric_sum/n).tolist()
        return dict(zip(('weighted_squared_deficit_per_foot_m2','actual_centroid_speed_m_s',
            'clearance_deficit_m','actual_sole_clearance_m','whole_robot_cost_divided_by_two'),values),
            actual_floor_slip_last_step=float(self.terms[0].mean()),
            low_moving_foot_cost_last_step=float(self.terms[1].mean()),foot_samples=int(self.samples))


def install(env,urdf,treatment=False,config=ClearanceConfig()):
    if abs(env.dt-.02)>1e-8 or env.num_obs!=1432:raise ValueError('Expected50Hz/1432 actor')
    if env.cfg.terrain.mesh_type!='plane':raise ValueError('Floor proxy requires flat plane')
    if any(env.gym.get_actor_count(e)!=1 for e in env.envs):raise ValueError('Expected single robot, no task props')
    body_names=env.gym.get_actor_rigid_body_names(env.envs[0],env.actor_handles[0])
    if [body_names[i] for i in env.feet_indices.tolist()]!=['left_ankle_roll_link','right_ankle_roll_link']:
        raise ValueError('Physical feet body order mismatch')
    state=ClearanceState(env,urdf,config)
    original_reward,original_reset=env.compute_reward,env.reset_idx
    if float(env.reward_scales['feet_slip'])>=0:raise ValueError('Expected source slip penalty')
    if treatment:
        env.reward_functions[env.reward_names.index('feet_slip')]=lambda:state.terms[0]
        name='low_moving_foot'
        env.reward_names.append(name);env.reward_functions.append(lambda:state.terms[1])
        env.reward_scales[name]=config.reward_coefficient*env.dt
        env.episode_sums[name]=torch.zeros(env.num_envs,device=env.device)
    def compute_reward():
        state.advance();original_reward()
    def reset(ids,*args,**kwargs):
        original_reset(ids,*args,**kwargs);state.reset(ids)
    env.compute_reward,env.reset_idx=compute_reward,reset
    return state


def gpu_contract_checks(env,state):
    """Real GPU tensor/geometry checks before the disposable optimizer smoke."""
    # Upstream reset simulates then refreshes rigid-body state only. Its root
    # and DOF copy buffers can still describe the pre-simulation instant. Refresh
    # all three copies without simulating before comparing their geometry.
    k=state.kinematics;rb=env.rigid_body_states[:,env.feet_indices]
    before_actual=k.world_soles(rb[...,:3],rb[...,3:7])
    before_fk,_=k.reference(env.root_states[:,:3],env.root_states[:,3:7],env.dof_pos)
    before_error=float((before_actual-before_fk).abs().max())
    sim_time_before=float(env.gym.get_sim_time(env.sim))
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    sim_time_after=float(env.gym.get_sim_time(env.sim))
    if sim_time_after!=sim_time_before:raise ValueError('Tensor refresh changed simulation time')
    rb=env.rigid_body_states[:,env.feet_indices]
    actual=k.world_soles(rb[...,:3],rb[...,3:7])
    fk,_=k.reference(env.root_states[:,:3],env.root_states[:,3:7],env.dof_pos)
    geometry_error=float((actual-fk).abs().max())
    if geometry_error>2e-4:raise ValueError('Rigid-body link pose/FK mismatch: '+str(geometry_error))
    current=actual[:1].clone();current[...,2]-=(current[...,2]-k.radii).min(-1).values[...,None]
    previous=current.clone();previous[:,0,:,0]-=.02
    cost,terms,_,_=sole_geometry_cost(current,previous,k.radii,.02)
    stationary=sole_geometry_cost(current,current,k.radii,.02)[0]
    raised=current.clone();raised[...,2]+=.10
    high=sole_geometry_cost(raised,raised+(previous-current),k.radii,.02)[0]
    expected=-torch.expm1(-torch.tensor(.065**2,device=env.device)*torch.tanh(torch.tensor(2.,device=env.device))/.05)
    if not torch.allclose(cost,expected[None],atol=2e-6):raise ValueError('GPU clearance formula mismatch')
    if float(terms[0,1])!=0 or float(stationary)!=0 or float(high)!=0:raise ValueError('GPU isolation/stand/high-foot contract')
    scale=float(env.reward_scales.get('low_moving_foot',0.))
    if scale and abs(scale+.02)>1e-8:raise ValueError('Clearance reward dt scaling mismatch')
    return dict(success=True,device=str(cost.device),num_envs=env.num_envs,actor_inputs=env.num_obs,
        initial_unsynchronized_FK_max_abs_m=before_error,
        sim_time_before_refresh=sim_time_before,sim_time_after_refresh=sim_time_after,
        refresh_advanced_simulation=False,
        actual_sphere_vs_root_joint_FK_max_abs_m=geometry_error,moving_floor_cost=float(cost),
        other_foot_numerator=float(terms[0,1]),stationary_cost=float(stationary),high_foot_cost=float(high),
        denominator=.05,height_target_m=.065,reward_scale=scale,dt_applied_exactly_once=True)

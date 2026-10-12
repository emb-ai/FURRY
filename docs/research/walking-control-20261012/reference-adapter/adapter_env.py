"""Isaac Gym reference adapter around an immutable TWIST2 tracker.

Three explicit modes: recorded motion, stopped-operator position catch-up,
stopped-operator turn. Training motions remain disjoint from validation.
"""
import torch
from adapter_core import (BodyKinematics,read_camera,qrot,qmul,inverse,angles,
                          yaw_quat,camera_servo,corrected_joints,tracker_observation)
from sole_kinematics import SoleKinematics


class AdapterEnv:
    def __init__(self,env,tracker,urdf,scene,seed=31):
        self.e=env;self.tracker=tracker.eval();self.device=env.device
        for p in tracker.parameters():p.requires_grad_(False)
        self.num_envs=n=env.num_envs;self.dt=env.dt
        if abs(self.dt-.02)>1e-8:raise ValueError('50Hz required')
        names=list(env.gym.get_actor_dof_names(env.envs[0],env.actor_handles[0]))
        expected=[s+'_'+j+'_joint' for s in ('left','right') for j in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
        if names[:12]!=expected:raise ValueError('Leg order mismatch')
        props=env.gym.get_actor_dof_properties(env.envs[0],env.actor_handles[0])
        self.lower=torch.tensor(props['lower'].copy(),device=self.device)
        self.upper=torch.tensor(props['upper'].copy(),device=self.device)
        self.body_names=list(env.gym.get_actor_rigid_body_names(env.envs[0],env.actor_handles[0]))
        self.camera_body,camera_offset=read_camera(scene)
        self.camera_id=self.body_names.index(self.camera_body)
        self.offset=torch.tensor(camera_offset,device=self.device)
        self.key_names=list(env.cfg.motion.key_bodies)
        self.fk=BodyKinematics(urdf,names,self.key_names+[self.camera_body],self.device)
        self.sole=SoleKinematics(urdf,names,self.device)
        self.rng=torch.Generator(device=self.device).manual_seed(seed)
        zeros=lambda *shape:torch.zeros(*shape,device=self.device)
        self.history=zeros(n,10,127);self.last_action=zeros(n,29)
        self.delta=zeros(n,12);self.previous_delta=zeros(n,12);self.delta_velocity=zeros(n,12)
        self.mode=torch.zeros(n,dtype=torch.long,device=self.device)
        self.goal_offset=zeros(n,3);self.goal_yaw=zeros(n)
        self.raw={};self.last_metrics={};self.episode_steps=0
        self.original_update=env._update_ref_motion
        original_times,original_reset,original_reward=env._get_motion_times,env.reset_idx,env.compute_reward
        def times(env_ids=None):
            t=original_times(env_ids)
            mode=self.mode if env_ids is None else self.mode[env_ids]
            offset=env._motion_time_offsets if env_ids is None else env._motion_time_offsets[env_ids]
            return torch.where(mode>0,offset,t)
        def update():
            self.original_update();self.capture_raw();self.apply_reference()
        def reset(ids,*args,**kwargs):
            original_reset(ids,*args,**kwargs)
            if len(ids):self.reset_buffers(ids)
        def reward():
            original_reward()
            # Frozen goals and physical outcomes are independent of corrected q.
            _,_,local_error,dyaw=self.navigation()
            distance=torch.linalg.vector_norm(local_error,dim=-1)
            progress=(self.before_distance-distance)/self.dt
            turn_progress=(self.before_yaw-dyaw.abs())/self.dt
            regularity=(self.delta/.5).square().mean(-1)
            smoothness=(self.delta-self.previous_delta).square().mean(-1)
            navigation=2*progress+turn_progress-.5*distance-.25*dyaw.abs()
            extra=navigation-.2*regularity-2*smoothness
            env.rew_buf+=self.dt*extra
            self.last_metrics={'distance_m':distance.detach(),'yaw_error_rad':dyaw.abs().detach(),
                'progress_m_s':progress.detach(),'turn_progress_rad_s':turn_progress.detach(),
                'delta_rms_rad':self.delta.square().mean(-1).sqrt().detach(),
                'mode':self.mode.clone(),'failure':(env.reset_buf & ~env.time_out_buf).clone(),
                'reward_rate':env.rew_buf.detach().clone()/self.dt}
            if hasattr(self,'clearance'):
                self.last_metrics['low_moving_cost']=self.clearance.terms[1].clone()
                self.last_metrics['contact_slip_cost']=self.clearance.terms[0].clone()
        env._get_motion_times=times;env._update_ref_motion=update
        env.reset_idx=reset;env.compute_reward=reward
        # Velocity imitation must target the very same catch-up command sent to
        # the frozen tracker, not the uncorrected operator velocity (often zero).
        def linear_reward():
            xy,_,_,_=self.navigation()
            actual=qrot(inverse(env.root_states[:,3:7]),env.root_states[:,7:10])
            return torch.exp(-(actual[:,:2]-xy).square().sum(-1)-actual[:,2].square())
        def angular_reward():
            _,wz,_,_=self.navigation()
            actual=qrot(inverse(env.root_states[:,3:7]),env.root_states[:,10:13])
            return torch.exp(-(actual[:,2]-wz).square()-actual[:,:2].square().sum(-1))
        for name,fn in [('tracking_root_linear_vel',linear_reward),('tracking_root_angular_vel',angular_reward)]:
            env.reward_functions[env.reward_names.index(name)]=fn
        original_termination=env.check_termination
        def termination():
            original_termination()
            # A finite eight-second correction task; reference may be held.
            limit=(self.mode>0)&(env.episode_length_buf*self.dt>=8.)
            failed=env.reset_buf & ~env.time_out_buf
            env.time_out_buf|=limit & ~failed;env.reset_buf|=limit
        env.check_termination=termination
        self.reset_buffers(torch.arange(n,device=self.device))
        update()

    def reset_buffers(self,ids):
        self.history[ids]=0;self.last_action[ids]=0;self.delta[ids]=0
        self.previous_delta[ids]=0;self.delta_velocity[ids]=0
        # 50% real trajectories, 25% held pose + position, 25% held pose + turn.
        u=torch.rand(len(ids),device=self.device,generator=self.rng)
        self.mode[ids]=torch.where(u<.5,0,torch.where(u<.75,1,2))
        xy=(torch.rand(len(ids),2,device=self.device,generator=self.rng)*2-1)*.6
        self.goal_offset[ids,:2]=xy*(self.mode[ids]==1)[:,None]
        self.goal_offset[ids,2]=0
        self.goal_yaw[ids]=(torch.rand(len(ids),device=self.device,generator=self.rng)*2-1)*1.0*(self.mode[ids]>0)

    def capture_raw(self):
        e=self.e
        for key in ('root_pos','root_rot','root_vel','root_ang_vel','dof_pos','dof_vel'):
            self.raw[key]=getattr(e,'_ref_'+key).clone()
        held=self.mode>0
        self.raw['root_pos']+=self.goal_offset
        self.raw['root_rot']=qmul(yaw_quat(self.goal_yaw),self.raw['root_rot'])
        for key in ('root_vel','root_ang_vel','dof_vel'):self.raw[key][held]=0
        poses,rots=self.fk(self.raw['root_pos'],self.raw['root_rot'],self.raw['dof_pos'])
        self.goal_pos=poses[:,-1]+qrot(rots[:,-1],self.offset)
        self.goal_quat=rots[:,-1]

    def apply_reference(self):
        e=self.e
        for key,value in self.raw.items():getattr(e,'_ref_'+key).copy_(value)
        e._ref_dof_pos[:,:12]=torch.maximum(torch.minimum(
            e._ref_dof_pos[:,:12]+self.delta,self.upper[:12]),self.lower[:12])
        self.delta=e._ref_dof_pos[:,:12]-self.raw['dof_pos'][:,:12]
        self.delta_velocity=(self.delta-self.previous_delta)/self.dt
        e._ref_dof_vel[:,:12]+=self.delta_velocity
        poses,_=self.fk(e._ref_root_pos,e._ref_root_rot,e._ref_dof_pos)
        e._ref_body_pos[:,e._key_body_ids]=poses[:,:len(self.key_names)]

    def navigation(self):
        e=self.e;rb=e.rigid_body_states[:,self.camera_id]
        camera=rb[:,:3]+qrot(rb[:,3:7],self.offset)
        local_feedforward=qrot(inverse(self.raw['root_rot']),self.raw['root_vel'])
        local_feedforward[:,2]=0
        world_feedforward=qrot(self.raw['root_rot'],local_feedforward)
        local_angular=qrot(inverse(self.raw['root_rot']),self.raw['root_ang_vel'])
        return camera_servo(world_feedforward,local_angular[:,2],e.root_states[:,3:7],
                            self.goal_pos,self.goal_quat,camera,rb[:,3:7])

    def command(self,corrected=False):
        xy,wz,_,_=self.navigation();r=self.raw
        # One tensor is authoritative after joint-limit projection. Rebuilding
        # raw + (projected - raw) can differ by a float32 ULP near a limit.
        dof=self.e._ref_dof_pos if corrected else r['dof_pos']
        return torch.cat((xy,r['root_pos'][:,2:3],angles(r['root_rot'])[:,:2],wz[:,None],dof),-1)

    def proprio(self):
        e=self.e;q=e.root_states[:,3:7]
        velocity=e.dof_vel.clone();velocity[:,[4,5,10,11]]=0
        return torch.cat((qrot(inverse(q),e.root_states[:,10:13])*.25,angles(q)[:,:2],
            e.dof_pos-e.default_dof_pos_all,velocity*.05,self.last_action),-1)

    def observation(self):
        command=self.command();obs,_=tracker_observation(command,self.proprio(),self.history)
        _,_,error,dyaw=self.navigation();e=self.e
        rb=e.rigid_body_states[:,e.feet_indices]
        h=(self.sole.world_soles(rb[:,:,:3],rb[:,:,3:7])[...,2]-self.sole.radii).min(-1).values
        force=(e.contact_forces[:,e.feet_indices,2]/350).clamp(0,2)
        v=qrot(inverse(e.root_states[:,3:7]),e.root_states[:,7:10])
        extras=torch.cat((error,torch.sin(dyaw)[:,None],torch.cos(dyaw)[:,None],
            h,force,v,self.delta/.5,(self.raw['dof_vel'][:,:12]/5).clamp(-2,2)),-1)
        return torch.cat((((obs-self.tracker.mean)/self.tracker.divisor).clamp(-10,10),extras),-1)

    @torch.no_grad()
    def step(self,action):
        e=self.e
        self.previous_delta=self.delta.clone()
        _,self.delta=corrected_joints(self.raw['dof_pos'],self.delta,action,
                                     self.lower,self.upper)
        self.delta_velocity=(self.delta-self.previous_delta)/self.dt
        self.apply_reference()
        command=self.command(corrected=True)
        if not torch.equal(command[:,6:18],e._ref_dof_pos[:,:12]):
            raise ValueError('Actor/reward corrected reference mismatch')
        obs,current=tracker_observation(command,self.proprio(),self.history)
        self.history=torch.cat((self.history[:,1:],current[:,None]),1)
        self.last_action=self.tracker(obs)
        _,_,error,dyaw=self.navigation()
        self.before_distance=torch.linalg.vector_norm(error,dim=-1);self.before_yaw=dyaw.abs()
        _,_,reward,done,info=e.step(self.last_action)
        # reset_idx advances simulation and refreshes only rigid bodies upstream.
        # Synchronize all read buffers without another physics step.
        e.gym.refresh_actor_root_state_tensor(e.sim);e.gym.refresh_dof_state_tensor(e.sim)
        e.gym.refresh_rigid_body_state_tensor(e.sim)
        # New episode reference is sampled in reset_idx; keep next observation
        # and this next step's reward source synchronized after a reset.
        self.original_update();self.capture_raw();self.apply_reference()
        next_obs=self.observation()
        if not torch.isfinite(next_obs).all() or not torch.isfinite(reward).all():
            raise FloatingPointError('Nonfinite adapter transition')
        return next_obs,reward,done,info

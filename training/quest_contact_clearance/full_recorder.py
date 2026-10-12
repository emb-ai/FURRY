"""Complete evaluation states and exact contact diagnostics, no state mutation."""
import json
from pathlib import Path
import numpy as np
import mujoco


def instrument(Base, directory):
    class Recorded(Base):
        def reset(self,*args,**kwargs):
            obs=super().reset(*args,**kwargs)
            self.record_rows=[];self.contact_impulse=np.zeros(2)
            return obs

        def contact_sample(self):
            result=super().contact_sample()
            if not hasattr(self,'contact_impulse'): return result
            m,d=self.model,self.contact_data
            impulse=np.zeros(2);f=np.zeros(6)
            cmd=self.motion['cmd'][self.k,:2]
            q=self.motion['q'][self.k,3:7]
            yaw=np.arctan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
            direction=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])@cmd
            norm=np.linalg.norm(direction)
            if norm>.05:
                direction/=norm
                for i in range(d.ncon):
                    c=d.contact[i];g1,g2=int(c.geom1),int(c.geom2)
                    if self.floor not in (g1,g2):continue
                    body=int(m.geom_bodyid[g2 if g1==self.floor else g1]);sides=np.flatnonzero(self.feet==body)
                    if not len(sides):continue
                    mujoco.mj_contactForce(m,d,i,f)
                    world=c.frame.reshape(3,3).T@f[:3]*(1 if g1==self.floor else -1)
                    impulse[sides[0]]+=max(0.,-world[:2]@direction)*m.opt.timestep
            self.contact_impulse+=impulse
            return result

        def step(self,action):
            actor_input=self.last_obs.copy()
            command=actor_input[:35].copy()
            history=actor_input[127:1397].reshape(10,127).copy()
            policy_tick=self.steps%self.policy_stride==0
            input_time_s=self.steps*self.dt
            self.contact_impulse=np.zeros(2)
            obs,reward,done,info=super().step(action)
            contacts=[];force=np.zeros(6);jac=np.zeros((3,self.model.nv))
            for i in range(self.contact_data.ncon):
                c=self.contact_data.contact[i];g1,g2=int(c.geom1),int(c.geom2)
                if self.floor not in (g1,g2):continue
                body=int(self.model.geom_bodyid[g2 if g1==self.floor else g1]);sides=np.flatnonzero(self.feet==body)
                if not len(sides):continue
                mujoco.mj_contactForce(self.model,self.contact_data,i,force)
                mujoco.mj_jac(self.model,self.contact_data,jac,None,c.pos,body)
                world=c.frame.reshape(3,3).T@force[:3]*(1 if g1==self.floor else -1)
                contacts.append(np.r_[sides[0],c.pos,jac@self.contact_data.qvel,world,max(0.,force[0])])
            self.record_rows.append(dict(qpos=self.data.qpos.copy(),qvel=self.data.qvel.copy(),
                reference_qpos=self.ref.qpos.copy(),command=command,action=np.asarray(action).copy(),
                actor_history=history,actor_input=actor_input,policy_tick=policy_tick,
                input_time_s=input_time_s,state_time_s=self.steps*self.dt,
                foot_position=self.data.xpos[self.feet].copy(),
                foot_quaternion=self.data.xquat[self.feet].copy(),
                normal_force=self.contact_force.copy(),braking_impulse=self.contact_impulse.copy(),
                contacts=np.asarray(contacts).reshape(-1,11)))
            if done:self.flush_record()
            return obs,reward,done,info

        def flush_record(self):
            directory_path=Path(directory);directory_path.mkdir(parents=True,exist_ok=True)
            clip=str(self.motion['meta']['id']).replace('/','_')
            # evaluate() sets explicit seed, avoiding overwrite on the second trial.
            prefix=directory_path/(clip+'__seed'+str(self.evaluation_seed))
            arrays={k:np.asarray([r[k] for r in self.record_rows]) for k in self.record_rows[0] if k!='contacts'}
            sizes=[len(r['contacts']) for r in self.record_rows]
            arrays['contact_offsets']=np.r_[0,np.cumsum(sizes)]
            arrays['contacts']=np.concatenate([r['contacts'] for r in self.record_rows])
            np.savez_compressed(str(prefix)+'.npz',**arrays)
            Path(str(prefix)+'.json').write_text(json.dumps(dict(schema=1,trace_hz=100,
                physics_hz=1000,policy_hz=100/self.policy_stride,seed=self.evaluation_seed,
                contact_columns=['side','x','y','z','vx','vy','vz','fx','fy','fz','normal_force'],
                braking_impulse_units='N s per 10 ms trace sample',
                sample_timing='actor_input is the held pre-step policy input; qpos/qvel/reference/contacts are post-step; policy_tick distinguishes new inference',
                provenance='fresh deterministic evaluator rollout of saved weights, not historical saved-state replay'),indent=2))
    return Recorded

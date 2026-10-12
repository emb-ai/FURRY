"""Empirically identify Isaac tensor linear velocity origin on an offset-COM body."""
import argparse
from pathlib import Path
import json
import isaacgym
from isaacgym import gymapi, gymtorch
import torch
from contact_clearance import rotate


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    urdf = output/'probe.urdf'
    urdf.write_text('''<robot name="origin_probe"><link name="body">
<inertial><origin xyz="0.12 0 0"/><mass value="1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/></inertial>
<collision><geometry><sphere radius="0.05"/></geometry></collision>
</link></robot>''')
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams(); params.dt = .001
    params.up_axis = gymapi.UP_AXIS_Z; params.gravity = gymapi.Vec3(0, 0, 0)
    params.use_gpu_pipeline = True; params.physx.use_gpu = True
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
    if sim is None: raise RuntimeError('PhysX unavailable')
    try:
        options = gymapi.AssetOptions(); options.disable_gravity = True
        asset = gym.load_asset(sim, str(output), urdf.name, options)
        env = gym.create_env(sim, gymapi.Vec3(-1,-1,-1), gymapi.Vec3(1,1,1), 1)
        pose = gymapi.Transform(); pose.p = gymapi.Vec3(0, 0, 2)
        actor = gym.create_actor(env, asset, pose, 'probe', 0, 0)
        gym.prepare_sim(sim)
        root = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        rb = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
        gym.refresh_actor_root_state_tensor(sim)
        root[:, 7:10] = torch.tensor([.3,.1,0.],device='cuda')
        root[:, 10:13] = torch.tensor([0.,2.,0.],device='cuda')
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root))
        com_property = gym.get_actor_rigid_body_properties(env, actor)[0].com
        com = torch.tensor([[com_property.x,com_property.y,com_property.z]],device='cuda')
        states = []
        for _ in range(201):
            gym.simulate(sim);gym.fetch_results(sim,True);gym.refresh_rigid_body_state_tensor(sim)
            states.append(rb.clone())
        states = torch.stack(states).squeeze(1)
        measured = (states[2:, :3]-states[:-2, :3])/(2*params.dt)
        mid = states[1:-1]
        offset = rotate(mid[:,3:7],com.expand(len(mid),3))
        link_prediction = mid[:,7:10]
        com_prediction = mid[:,7:10]-torch.cross(mid[:,10:13],offset,dim=-1)
        errors = {name:float((pred-measured).square().mean().sqrt()) for name,pred in
                  [('link',link_prediction),('com',com_prediction)]}
        selected = min(errors,key=errors.get)
        passed = errors[selected] < .003 and max(errors.values()) > .05
        report = dict(passed=passed, velocity_origin=selected, world_frame=True, rms_m_s=errors,
                      com_local=com.cpu().tolist(), dt=params.dt, samples=len(mid),
                      method='central finite difference of rotating offset-COM link position')
        (output/'result.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
        if not passed: raise RuntimeError('Ambiguous velocity origin')
    finally:
        gym.destroy_sim(sim)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output.resolve())

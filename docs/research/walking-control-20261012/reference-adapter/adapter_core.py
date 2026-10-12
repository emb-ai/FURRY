"""Causal lower-body reference correction; no motor commands or pose overrides.

All quaternions use XYZW. Joint and camera geometry is read from model assets.
The 12 outputs modify the reference, never the simulated robot state.
"""
import math
import xml.etree.ElementTree as ET
import torch
from sole_kinematics import qmul, qrot, rpy_quat, yaw


def inverse(q):
    return torch.cat((-q[..., :3], q[..., 3:]), -1)


def angles(q):
    x,y,z,w=q.unbind(-1)
    return torch.stack((torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),
                        torch.asin((2*(w*y-z*x)).clamp(-1,1)),yaw(q)), -1)


def yaw_quat(a):
    return torch.stack((torch.zeros_like(a),torch.zeros_like(a),torch.sin(a/2),torch.cos(a/2)), -1)


def norm_clip(x, limit):
    return x*(limit/torch.linalg.vector_norm(x,dim=-1,keepdim=True).clamp(min=limit))


def camera_servo(world_velocity, world_yaw_rate, actual_root_quat,
                 goal_pos, goal_quat, actual_pos, actual_quat):
    """Native CameraPoseServo gains/caps, expressed with torso-forward +X."""
    error=goal_pos-actual_pos;error=error.clone();error[:,2]=0
    local_error=qrot(inverse(actual_root_quat),error)
    correction=norm_clip(local_error[:,:2],.4)
    local=qrot(inverse(actual_root_quat),world_velocity)
    xy=norm_clip(local[:,:2]+correction,.8)
    dyaw=yaw(goal_quat)-yaw(actual_quat)
    dyaw=torch.atan2(torch.sin(dyaw),torch.cos(dyaw))
    wz=(world_yaw_rate+(3*dyaw).clamp(-1,1)).clamp(-1.2,1.2)
    return xy,wz,local_error[:,:2],dyaw


class BodyKinematics:
    """FK of requested URDF links; shared ancestors evaluated once."""
    def __init__(self,urdf,dof_names,body_names,device):
        xml=ET.parse(urdf).getroot()
        edges={j.find('child').get('link'):j for j in xml.findall('joint')}
        needed=set()
        for name in body_names:
            while name!='pelvis':
                if name not in edges:raise ValueError('Unknown URDF link '+name)
                needed.add(name);name=edges[name].find('parent').get('link')
        order=[];done={'pelvis'}
        while needed:
            ready=sorted(n for n in needed if edges[n].find('parent').get('link') in done)
            if not ready:raise ValueError('Cyclic/disconnected URDF')
            for name in ready:
                j=edges[name];origin=j.find('origin')
                xyz=list(map(float,origin.get('xyz','0 0 0').split())) if origin is not None else [0,0,0]
                rpy=list(map(float,origin.get('rpy','0 0 0').split())) if origin is not None else [0,0,0]
                idx=None;axis=[0,0,0]
                if j.get('type')!='fixed':
                    if j.get('type') not in ('revolute','continuous'):raise ValueError(j.get('type'))
                    idx=list(dof_names).index(j.get('name'));axis=list(map(float,j.find('axis').get('xyz').split()))
                tensor=lambda x:torch.tensor(x,dtype=torch.float32,device=device)
                order.append((name,j.find('parent').get('link'),tensor(xyz),tensor(rpy_quat(rpy)),idx,tensor(axis)))
            done.update(ready);needed.difference_update(ready)
        self.order=order;self.names=list(body_names)

    def __call__(self,root_pos,root_quat,dof):
        poses={'pelvis':(root_pos,root_quat)}
        for name,parent,xyz,fixed,idx,axis in self.order:
            p,q=poses[parent];p=p+qrot(q,xyz);q=qmul(q,fixed.expand_as(q))
            if idx is not None:
                half=dof[:,idx]/2
                q=qmul(q,torch.cat((torch.sin(half)[:,None]*axis,torch.cos(half)[:,None]),-1))
            poses[name]=(p,q)
        return torch.stack([poses[n][0] for n in self.names],1),torch.stack([poses[n][1] for n in self.names],1)


def read_camera(xml):
    tree=ET.parse(xml).getroot()
    for body in tree.iter('body'):
        for camera in body.findall('camera'):
            if camera.get('name')=='ego':
                if camera.get('xyaxes')!='0 -1 0 0 0 1':
                    raise ValueError('Expected ego camera facing body +X')
                return body.get('name'),list(map(float,camera.get('pos').split()))
    raise ValueError('Missing ego camera')


def corrected_joints(raw, previous, action, lower, upper, amplitude=.5, alpha=.2):
    requested=(1-alpha)*previous+alpha*amplitude*torch.tanh(action)
    legs=torch.maximum(torch.minimum(raw[:,:12]+requested,upper[:12]),lower[:12])
    result=raw.clone();result[:,:12]=legs
    return result,legs-raw[:,:12]


def tracker_observation(command,proprio,history):
    current=torch.cat((command,proprio),-1)
    return torch.cat((current,history.flatten(1),command),-1),current

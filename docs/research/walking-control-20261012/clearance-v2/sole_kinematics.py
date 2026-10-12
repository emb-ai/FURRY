"""URDF physical sole geometry; unchanged FK from verified v1 adapter."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import torch


def qmul(a,b):
    av,bv=a[...,:3],b[...,:3]
    return torch.cat((a[...,3:]*bv+b[...,3:]*av+torch.cross(av,bv,dim=-1),
                      a[...,3:]*b[...,3:]-(av*bv).sum(-1,keepdim=True)),-1)


def qrot(q,v):
    # Retain the scalar quaternion component while broadcasting XYZ vectors.
    quat,vec=q,v
    xyz,vec=torch.broadcast_tensors(quat[...,:3],vec)
    t=2*torch.cross(xyz,vec,dim=-1)
    return vec+quat[...,3:]*t+torch.cross(xyz,t,dim=-1)


def yaw(q):
    x,y,z,w=q.unbind(-1)
    return torch.atan2(2*(w*z+x*y),w*w+x*x-y*y-z*z)


def rpy_quat(rpy):
    r,p,y=[v/2 for v in rpy];cr,sr,cp,sp,cy,sy=math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    return [sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy,cr*cp*cy+sr*sp*sy]


class SoleKinematics:
    """Two leg-chain FK from the actual sphere-sole URDF, XYZW quaternions."""
    def __init__(self,urdf,dof_names,device='cpu',dtype=torch.float32):
        tree=ET.parse(urdf).getroot();joints={j.find('child').get('link'):j for j in tree.findall('joint')}
        self.chains=[];points=[];radii=[];self.foot_names=['left_ankle_roll_link','right_ankle_roll_link']
        self.tensor=lambda x:torch.tensor(x,device=device,dtype=dtype)
        for foot in self.foot_names:
            chain=[];link=foot
            while link!='pelvis':
                if link not in joints:raise ValueError('Foot chain does not lead to pelvis')
                j=joints[link];origin=j.find('origin');xyz=[0.,0.,0.];rpy=xyz
                if origin is not None:
                    xyz=list(map(float,origin.get('xyz','0 0 0').split()));rpy=list(map(float,origin.get('rpy','0 0 0').split()))
                idx=None;axis=[0.,0.,0.]
                if j.get('type')!='fixed':
                    if j.get('type') not in ['revolute','continuous']:raise ValueError('Unsupported leg joint')
                    idx=list(dof_names).index(j.get('name'));axis=list(map(float,j.find('axis').get('xyz').split()))
                chain.append((self.tensor(xyz),self.tensor(rpy_quat(rpy)),idx,self.tensor(axis)))
                link=j.find('parent').get('link')
            self.chains.append(chain[::-1]);p=[];r=[]
            body=tree.find("link[@name='%s']"%foot)
            for collision in body.findall('collision'):
                sphere=collision.find('geometry/sphere')
                if sphere is None:raise ValueError('Phase training requires sphere-only foot collisions')
                origin=collision.find('origin');p.append(list(map(float,origin.get('xyz','0 0 0').split())) if origin is not None else [0,0,0]);r.append(float(sphere.get('radius')))
            if not p:raise ValueError('Missing physical sole spheres')
            points.append(p);radii.append(r)
        if len(points[0])!=len(points[1]):raise ValueError('Unequal foot sphere count')
        self.points,self.radii=self.tensor(points),self.tensor(radii)

    def world_soles(self,pos,quat):
        quat=quat/torch.linalg.vector_norm(quat,dim=-1,keepdim=True).clamp(min=1e-8)
        return pos[...,None,:]+qrot(quat[...,None,:],self.points[None])

    def reference(self,root_pos,root_quat,dof):
        poses=[];rots=[];root_quat=root_quat/torch.linalg.vector_norm(root_quat,dim=-1,keepdim=True).clamp(min=1e-8)
        for chain in self.chains:
            p,q=root_pos,root_quat
            for origin,fixed,index,axis in chain:
                p=p+qrot(q,origin);q=qmul(q,fixed.expand_as(q))
                if index is not None:
                    half=.5*dof[:,index];rotation=torch.cat((axis[None]*torch.sin(half)[:,None],torch.cos(half)[:,None]),-1)
                    q=qmul(q,rotation)
            poses.append(p);rots.append(q)
        pos,quat=torch.stack(poses,1),torch.stack(rots,1)
        return self.world_soles(pos,quat),yaw(quat)

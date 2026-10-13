"""Offline support projection for the Quest ankle-only adaptation.

World XY toe anchors are acquired once per proposed source stance. The
floating translation and 12 leg joints may move; root orientation, waist,
arms and wrists retain the frozen GMR reference. Failed solves are reported
and prevent a reference being marked ready. This is our adaptation, not the
unpublished DexWeave retargeter or a claim of dynamic feasibility.
"""
import argparse,json
from pathlib import Path
import numpy as np
import mujoco as mj
from scipy.optimize import minimize
from .reference import Geometry,angular_velocity,sha

class Projector:
    def __init__(self,geometry):
        self.g=geometry;self.qids=np.r_[0:3,7:19];self.dids=np.r_[0:3,geometry.dadr[:12]]
        self.anchors=[None,None];self.previous_delta=np.zeros(15)
    def solve(self,original,stance):
        g=self.g;g.set(original)
        before=np.array([g.data.xpos[b]+g.data.xmat[b].reshape(3,3)@np.array([.14,0,-.03]) for b in g.feet])
        for s in range(2):
            if not stance[s]:self.anchors[s]=None
            elif self.anchors[s] is None:self.anchors[s]=before[s,:2].copy()
        active=np.flatnonzero(stance)
        # Release correction smoothly when both feet leave support.
        x0=original[self.qids].copy();weights=np.r_[np.full(3,1/.05**2),np.full(12,1/.2**2)]
        cache_x=None;cache=None
        def evaluate(x):
            nonlocal cache_x,cache
            if cache_x is not None and np.array_equal(x,cache_x):return cache
            q=original.copy();q[self.qids]=x;g.set(q);mj.mj_comPos(g.model,g.data)
            eq=[];ej=[];h=[];hj=[]
            for s,b in enumerate(g.feet):
                if s in active:
                    point=g.data.xpos[b]+g.data.xmat[b].reshape(3,3)@np.array([.14,0,-.03])
                    jac=np.zeros((3,g.model.nv));mj.mj_jac(g.model,g.data,jac,None,point,b)
                    eq.extend(point[:2]-self.anchors[s]);ej.extend(jac[:2,self.dids])
                for geom,local,radius in g.pads[s]:
                    point=g.data.geom_xpos[geom]+g.data.geom_xmat[geom].reshape(3,3)@local
                    jac=np.zeros((3,g.model.nv));mj.mj_jac(g.model,g.data,jac,None,point,b)
                    h.append(point[2]-radius);hj.append(jac[2,self.dids])
            cache_x=x.copy();cache=(np.array(eq),np.asarray(ej).reshape(-1,15),np.array(h),np.array(hj))
            return cache
        def objective(x):
            delta=x-x0
            return .5*np.sum(weights*delta**2)+.05*np.sum(weights*(delta-self.previous_delta)**2)
        def gradient(x):
            delta=x-x0
            return weights*delta+.1*weights*(delta-self.previous_delta)
        bounds=[(None,None)]*3+list(map(tuple,g.ranges[:12]))
        constraints=[{'type':'ineq','fun':lambda x:evaluate(x)[2], 'jac':lambda x:evaluate(x)[3]}]
        if len(active):constraints.append({'type':'eq','fun':lambda x:evaluate(x)[0],'jac':lambda x:evaluate(x)[1]})
        result=minimize(objective,x0+self.previous_delta,method='SLSQP',jac=gradient,bounds=bounds,
                        constraints=constraints,options={'maxiter':60,'ftol':1e-9})
        eq,_,h,_=evaluate(result.x)
        error=float(abs(eq).max()) if len(eq) else 0.
        ok=bool(result.success and error<1e-5 and h.min()>-1e-6)
        q=original.copy();q[self.qids]=result.x
        self.previous_delta=result.x-x0
        return q,{'ok':ok,'anchor_error_m':error,'clearance_m':float(h.min()),'iterations':int(result.nit),'message':result.message}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('references',type=Path);ap.add_argument('output',type=Path);ap.add_argument('--all',action='store_true');args=ap.parse_args()
    m=json.loads((args.references/'manifest.json').read_text());g=Geometry(m['model_path'])
    if sha(m['model_path'])!=m['model_sha256']:raise ValueError('Changed model')
    args.output.mkdir(parents=True,exist_ok=True);records=[]
    for c in m['clips']:
        if not args.all and c['id'] not in m['pilot_ids']:continue
        with np.load(args.references/c['file']) as src:arrays={k:src[k].copy() for k in src.files}
        q0=arrays['qpos'];stance=arrays['support'];solver=Projector(g);q=[];diagnostics=[]
        for row,mask in zip(q0,stance):
            qr,d=solver.solve(row,mask);q.append(qr);diagnostics.append(d)
        q=np.array(q);p,r,h,toe=g.fk(q)
        valid=stance[1:]&stance[:-1]
        before=np.linalg.norm(np.diff(arrays['toe_pos_w'][:,:,:2],axis=0),axis=2)/.02
        after=np.linalg.norm(np.diff(toe[:,:,:2],axis=0),axis=2)/.02
        info={'id':c['id'],'source_sha256':c['sha256'],'split':c['split'],'frames':len(q),
            'failed_frames':[i for i,d in enumerate(diagnostics) if not d['ok']],
            'anchor_error_max_m':max(d['anchor_error_m'] for d in diagnostics),
            'floor_min_m':float(h.min()),'root_correction_max_m':float(np.linalg.norm(q[:,:3]-q0[:,:3],axis=1).max()),
            'joint_correction_max_rad':float(abs(q[:,7:]-q0[:,7:]).max()),
            'upper_joint_difference_max_rad':float(abs(q[:,19:]-q0[:,19:]).max()),
            'support_toe_speed_before_m_s':float(before[valid].mean()) if valid.any() else None,
            'support_toe_speed_after_m_s':float(after[valid].mean()) if valid.any() else None,
            'max_joint_speed_before_rad_s':float(abs(np.diff(q0[:,7:],axis=0)/.02).max()),
            'max_joint_speed_after_rad_s':float(abs(np.diff(q[:,7:],axis=0)/.02).max())}
        info['ready_solver']=not info['failed_frames']
        arrays.update(qpos=q,joint_pos=q[:,7:],joint_vel=np.gradient(q[:,7:],.02,axis=0),body_pos_w=p,body_quat_w=r,
                      body_lin_vel_w=np.gradient(p,.02,axis=0),body_ang_vel_w=angular_velocity(r,.02),floor_clearance=h,toe_pos_w=toe)
        dst=args.output/c['file'];dst.parent.mkdir(exist_ok=True);np.savez_compressed(dst,**arrays)
        info['file']=c['file'];info['sha256']=sha(dst);records.append(info)
        (args.output/'report.json').write_text(json.dumps({'schema':'furry-quest-support-projection-v1','source_manifest_sha256':sha(args.references/'manifest.json'),
            'status':'kinematic experiment; dynamic tracking not verified','clips':records},indent=2)+'\n')
        print(json.dumps(info),flush=True)

if __name__=='__main__':main()

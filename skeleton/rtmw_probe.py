"""RTMW3D RGB / aligned RGB-D probe; deliberately not a robot control source.

NPY: BGR uint8 image. NPZ: color (BGR), depth_m aligned to color,
intrinsics [fx, fy, cx, cy]. Stores all 133 2D points and monocular relative Z.
The RTMLib 3D return has crop-pixel X/Y, not XYZ metres: never transmit it as XYZ.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np


def monocular_prior(xy, scores, relative_z, measured, intrinsics):
    """Experimental current-frame 3D prior, anchored by metric RGB-D torso.

    Model Z's nominal metre scale is retained, NOT fitted to each noisy frame.
    Robust torso depth offset aligns the prediction to the camera. This is a
    prediction even at high confidence; consumers must never mark it measured.
    Reject mutually inconsistent depth anchors. No previous-frame hold.
    """
    if relative_z is None:return {}
    xy=np.asarray(xy);scores=np.asarray(scores);z=np.asarray(relative_z)
    anchors=[i for i in (5,6,11,12) if i in measured and measured[i]['conf']>=.55
             and np.isfinite(z[i])]
    if len(anchors)<3:return {}
    offsets=np.array([measured[i]['p'][2]-z[i] for i in anchors])
    offset=np.median(offsets)
    if np.max(np.abs(offsets-offset))>.25:return {}
    fx,fy,cx,cy=intrinsics
    result={}
    for i,((u,v),score,relative) in enumerate(zip(xy,scores,z)):
        depth=relative+offset
        if not np.isfinite([u,v,depth,score]).all() or score<.2 or not .2<depth<6:continue
        result[i]=[(float(u)-cx)*depth/fx,(float(v)-cy)*depth/fy,float(depth)]
    return result


def metric_points(xy, scores, depth, intrinsics):
    """Conservative depth sampling, no along-bone substitution or filling.

    A local mixed-depth patch is rejected. This is a surface landmark estimate,
    not the anatomical joint centre. No metric points outside the RGB-D image.
    """
    fx,fy,cx,cy=intrinsics
    if not np.isfinite(intrinsics).all() or fx<=0 or fy<=0:
        raise ValueError('Invalid intrinsics')
    h,w=depth.shape;result={}
    for i,((u,v),score) in enumerate(zip(xy,scores)):
        if not np.isfinite([u,v,score]).all() or score<.55 or not(0<=u<w and 0<=v<h):continue
        x,y=int(round(u)),int(round(v))
        patch=depth[max(0,y-3):min(h,y+4),max(0,x-3):min(w,x+4)]
        values=patch[np.isfinite(patch)&(patch>.2)&(patch<6)]
        if len(values)<max(5,.5*patch.size):continue
        z=float(np.median(values));mad=float(np.median(abs(values-z)))
        if mad>.03 or np.quantile(values,.9)-np.quantile(values,.1)>.12:continue
        result[i]={'p':[(float(u)-cx)*z/fx,(float(v)-cy)*z/fy,z],
                   'conf':float(np.clip(score,0,1)),'src':'window','depth_mad_m':mad}
    return result


def load_models(folder, provider):
    from rtmlib.tools import YOLOX, RTMPose3d
    import onnxruntime as ort
    det=next(Path(folder).rglob('*det*.onnx'),None)
    if det is None:
        candidates=[p for p in Path(folder).rglob('*.onnx') if p.name!='pose.onnx']
        if len(candidates)!=1:raise ValueError('Expected one extracted YOLOX ONNX detector')
        det=candidates[0]
    pose=Path(folder)/'pose.onnx'
    if provider not in ort.get_available_providers():raise ValueError('Provider unavailable: '+provider)
    detector=YOLOX(str(det),model_input_size=(640,640),backend='onnxruntime',device='cpu')
    estimator=RTMPose3d(str(pose),model_input_size=(288,384),backend='onnxruntime',device='cpu')
    # Bound CPU contention; report actual providers, including fallback.
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    for model in (detector,estimator):
        model.session=ort.InferenceSession(model.onnx_model,options,providers=[provider,'CPUExecutionProvider'] if provider!='CPUExecutionProvider' else [provider])
    return detector,estimator


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('input');ap.add_argument('--models',default='models/rtmw3d')
    ap.add_argument('--provider',default='CPUExecutionProvider')
    ap.add_argument('--runs',type=int,default=10);ap.add_argument('--output',required=True)
    args=ap.parse_args()
    loaded=np.load(args.input,allow_pickle=False)
    depth=intr=None
    if isinstance(loaded,np.lib.npyio.NpzFile):
        image=loaded['color'];depth=loaded['depth_m'];intr=loaded['intrinsics']
        if depth.shape!=image.shape[:2]:raise ValueError('Depth must be aligned to color')
    else:image=loaded
    detector,pose=load_models(args.models,args.provider)
    times=[];people=[]
    for n in range(args.runs+2):
        t=time.perf_counter();boxes=detector(image);d=time.perf_counter()
        people=[]
        # RTMLib defaults to the full frame on an empty box list: bypass explicitly.
        for box in boxes:
            xyz,score,_,xy=pose(image,bboxes=[box])
            p={'bbox':box.tolist(),'xy':xy[0].tolist(),'score':score[0].tolist(),
               'relative_z_model':xyz[0,:,2].tolist()}
            if depth is not None:p['depth_points']=metric_points(xy[0],score[0],depth,intr)
            people.append(p)
        end=time.perf_counter()
        if n>=2:times.append([(d-t)*1000,(end-d)*1000,(end-t)*1000])
    output={'provider':pose.session.get_providers(),'people':people,
            'timing_median_ms_det_pose_total':np.median(times,axis=0).tolist(),
            'timing_p95_ms_det_pose_total':np.quantile(times,.95,axis=0).tolist(),
            'has_depth':depth is not None,'runs':args.runs,
            'warning':'Repeated image benchmark, not a tracking quality/latency test. No automatic person selection or robot output.'}
    Path(args.output).write_text(json.dumps(output))
    print(json.dumps({k:v for k,v in output.items() if k!='people'},indent=2))
    print('Detected people:',len(people))

if __name__=='__main__':main()

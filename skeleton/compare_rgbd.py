"""Replay one RGB-D recording through one backend per process; no camera access.

Scores are NOT calibrated across networks. Export raw scores and test threshold
sensitivity before interpreting dropout as accuracy. No robot commands.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from rtmw_probe import load_models,metric_points,monocular_prior

def iou(a,b):
    low=np.maximum(a[:2],b[:2]);high=np.minimum(a[2:],b[2:])
    intersection=np.prod(np.maximum(0,high-low))
    return intersection/max(1e-9,np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('recording');ap.add_argument('backend',choices=['yolo','rtmw'])
    ap.add_argument('--models',default='models/rtmw3d');ap.add_argument('--yolo',default='models/yolo11s-pose.pt')
    ap.add_argument('--provider',default='CoreMLExecutionProvider');ap.add_argument('--stride',type=int,default=1)
    ap.add_argument('--output',required=True)
    ap.add_argument('--min-area-fraction',type=float,default=0,help='Explicit foreground selection gate; record value in report')
    args=ap.parse_args()
    if args.backend=='yolo':
        from ultralytics import YOLO
        model=YOLO(args.yolo)
    else:detector,model=load_models(args.models,args.provider)
    previous=None;count=0
    with Path(args.output).open('w') as out:
        for path in sorted(Path(args.recording).glob('frame-*.npz'))[::args.stride]:
            with np.load(path,allow_pickle=False) as frame:
                image=frame['color'];depth=frame['depth_m'];intr=frame['intrinsics']
                # Current recorder preserves distortion metadata. This probe only
                # supports rectified / zero-distortion pinhole frames.
                if 'distortion_coeffs' in frame and np.any(np.abs(frame['distortion_coeffs'])>1e-8):
                    raise ValueError('Nonzero lens distortion needs SDK deprojection')
                t=time.perf_counter()
                if args.backend=='yolo':
                    result=model.predict(image,imgsz=640,conf=.3,classes=0,device='mps',verbose=False)[0]
                    boxes=result.boxes.xyxy.cpu().numpy()
                else:boxes=detector(image)
                selected=None
                if len(boxes):
                    areas=np.prod(boxes[:,2:]-boxes[:,:2],axis=1)
                    eligible=np.flatnonzero(areas>=args.min_area_fraction*image.shape[0]*image.shape[1])
                    if len(eligible):
                        if previous is None:selected=int(eligible[np.argmax(areas[eligible])])
                        else:
                            overlap=np.array([iou(previous,boxes[i]) for i in eligible]);best=int(np.argmax(overlap))
                            if overlap[best]>.2:selected=int(eligible[best])
                row={'file':path.name,'t':int(frame['epoch_ms']),'seq':int(frame['sequence']),
                     'backend':args.backend,'min_area_fraction':args.min_area_fraction,'detected_people':len(boxes),'tracked':selected is not None}
                if selected is not None:
                    box=boxes[selected];previous=box.copy()
                    if args.backend=='yolo':
                        xy=result.keypoints.xy.cpu().numpy()[selected];score=result.keypoints.conf.cpu().numpy()[selected]
                        z=None
                    else:
                        xyz,scores,_,pixels=model(image,bboxes=[box]);xy=pixels[0];score=scores[0];z=xyz[0,:,2]
                    row.update(bbox=box.tolist(),xy=xy.tolist(),score=score.tolist(),relative_z_model=None if z is None else z.tolist())
                    row['inference_ms']=(time.perf_counter()-t)*1000
                    # Geometry for every positive detection; thresholds analysed offline.
                    depth_points=metric_points(xy,np.ones(len(score)),depth,intr)
                    for i,j in depth_points.items():j['conf']=float(np.clip(score[i],0,1))
                    row['depth_points']=depth_points
                    if z is not None:
                        row['monocular_prior_points']=monocular_prior(xy,score,z,depth_points,intr)
                out.write(json.dumps(row)+'\n');count+=1
                if count%50==0:print(args.backend,count,flush=True)
    print('Finished',count,flush=True)

if __name__=='__main__':main()

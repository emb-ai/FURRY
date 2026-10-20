from pathlib import Path
import json,numpy as np,torch
O=Path(__file__).resolve().parent;W=O.parent/'cup-homie-finetune-20261015/finetuned-weights/act';torch.set_num_threads(2);m=torch.jit.load(str(W/'act-inference.pt')).eval();s={k:{kk:np.array(v,dtype=np.float32) for kk,v in val.items()} for k,val in json.loads((W/'stats.json').read_text()).items()};prev=[]
for name in ['demo_0','demo_29','demo_58']:
 z=np.load(O/'render-input'/(name+'.npz'))
 for i in [0,len(z['actions'])//2]:prev.append(z['previous_command'][i])
x=(np.array(prev)-s['obs/previous_command']['mean'])/s['obs/previous_command']['std'];pred=[]
for file in ['renderer-check.npy','renderer-check-remote.npy']:
 im=np.load(O/file).transpose(0,3,1,2).astype(np.float32)/255;im=(im-np.array([.485,.456,.406],np.float32)[None,:,None,None])/np.array([.229,.224,.225],np.float32)[None,:,None,None]
 with torch.inference_mode():y=m(torch.from_numpy(im),torch.from_numpy(x)).numpy()[:,:8]*s['actions']['std']+s['actions']['mean']
 pred.append(y)
e=np.abs(pred[0]-pred[1]);report={'frames':6,'compared_execution_actions':8,'arms_difference_mae_deg':float(np.rad2deg(e[:,:,21:35]).mean()),'arms_difference_max_deg':float(np.rad2deg(e[:,:,21:35]).max()),'grip_mae':float(e[:,:,35:37].mean()),'gaze_mae_deg':float(np.rad2deg(e[:,:,37:39]).mean()),'caveat':'Paired image sensitivity at fixed inputs, not closed-loop success'};(O/'renderer-prediction-parity.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

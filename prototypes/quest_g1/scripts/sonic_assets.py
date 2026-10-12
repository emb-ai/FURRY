"""Validate and package pinned SONIC encoder, decoder and planner contracts."""
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import onnxruntime as ort
ROOT=Path(__file__).resolve().parents[1]
PROVENANCE=ROOT/'policies/sonic-vr3/provenance.json'

def package(row,assets,opts,checked_model):
    proof=json.loads(PROVENANCE.read_text())
    def verify(path,original):
        with open(path,'rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=proof['sha256'][original]:raise ValueError('Pinned SONIC digest mismatch: '+original)
        return digest
    files={}
    for prefix,original,kind in [('', 'model_encoder.onnx','sonic_encoder'),('secondary_', 'model_decoder.onnx','sonic_decoder')]:
        path=Path(row[prefix+'path']);digest=verify(path,original)
        checked_model(path,row['id'],kind,digest,opts)
        filename=f'policy-{row["id"]}-{kind}.onnx';shutil.copy2(path,assets/filename)
        files[prefix+'file']=filename;files[prefix+'sha256']=digest
    path=Path(row['planner_path']);digest=verify(path,'planner_sonic.onnx')
    s=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
    feed={'context_mujoco_qpos':np.zeros((1,4,36),np.float32),'target_vel':np.array([.01],np.float32),
          'mode':np.zeros(1,np.int64),'movement_direction':np.zeros((1,3),np.float32),'facing_direction':np.array([[1,0,0]],np.float32),
          'random_seed':np.array([11],np.int64),'has_specific_target':np.zeros((1,1),np.int64),'specific_target_positions':np.zeros((1,4,3),np.float32),
          'specific_target_headings':np.zeros((1,4),np.float32),'allowed_pred_num_tokens':np.array([[1]*6+[0]*5],np.int64),'height':np.array([-1],np.float32)}
    constants=json.loads((ROOT/'experiments/homie_sonic/sonic_constants.json').read_text())
    feed['context_mujoco_qpos'][0,:,2]=.793;feed['context_mujoco_qpos'][0,:,3]=1;feed['context_mujoco_qpos'][0,:,7:]=constants['home']
    q,n=s.run(None,feed)
    if q.shape!=(1,64,36) or n.shape!=(1,) or not 4<=n.item()<=64 or not np.isfinite(q).all():raise ValueError('Invalid SONIC planner output')
    filename=f'policy-{row["id"]}-planner.onnx';shutil.copy2(path,assets/filename)
    files.update(planner_file=filename,planner_sha256=digest)
    source=PROVENANCE.parent
    for name in ['sonic_motor_profile.txt','observation_config.yaml']:
        verify(source/name,name);shutil.copy2(source/name,assets/('sonic_observation_config.yaml' if name.endswith('.yaml') else name))
    for name in ['SOURCE_LICENSE.txt','MODEL_LICENSE.txt','NOTICE.txt']:
        shutil.copy2(source/name,assets/('SONIC_'+name))
    return files

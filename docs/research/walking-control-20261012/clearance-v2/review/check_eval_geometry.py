"""Independent 10/20ms evaluation FD and shadow-state geometry check."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np
import mujoco as mj

HERE=Path(__file__).resolve().parent;SOURCE=HERE.parent;sys.path.insert(0,str(SOURCE))
from clearance_evaluation import ClearanceRecorder
from sole_geometry import sole_clearance

import argparse
p=argparse.ArgumentParser();p.add_argument('--asset',required=True,type=Path);p.add_argument('--urdf',required=True,type=Path);args=p.parse_args()
asset=args.asset;urdf=args.urdf
m=mj.MjModel.from_xml_path(str(asset));d=mj.MjData(m);mj.mj_resetData(m,d)
names=[j.get('name') for j in ET.parse(urdf).getroot().findall('joint') if j.get('type') in ('revolute','continuous')]
qa=np.array([m.jnt_qposadr[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,n)] for n in names])
feet=np.array([mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,s+'_ankle_roll_link') for s in ('left','right')])
floor=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'floor')
q0=np.r_[d.qpos[:7],d.qpos[qa]];trajectory=np.tile(q0,(10,1));trajectory[:,0]+=np.arange(10)*.01
e=SimpleNamespace(model=m,data=d,feet=feet,floor=floor,qa=qa,k=0,motion={'q':trajectory})
rec=ClearanceRecorder(e);rec.reset();errors=[];records=[]
state_spec=mj.mjtState.mjSTATE_INTEGRATION
for k in range(1,10):
    e.k=k;d.qpos[:7]=trajectory[k,:7];d.qpos[qa]=trajectory[k,7:]
    # Deliberately leave the main FK cache stale; diagnostics must use scratch FK.
    before=np.zeros(mj.mj_stateSize(m,state_spec));mj.mj_getState(m,d,before,state_spec)
    values=rec.record();after=np.zeros_like(before);mj.mj_getState(m,d,after,state_spec)
    np.testing.assert_array_equal(before,after)
    independent=np.mean([sole_clearance(m,rec.actual,b,floor) for b in feet])
    assert abs(values['actual_sole_clearance_m']-independent)<1e-12
    for field in ('actual_foot_speed_m_s','reference_foot_speed_m_s','foot_directed_progress_m_s'):
        errors.append(abs(values[field]-1.));assert abs(values[field]-1.)<1e-12
    assert values['frozen_requested_foot_fraction']==0
    records.append(values)
result=dict(status='PASS',samples=9,history='First10ms then trailing20ms; constant velocity1m/s',
    velocity_max_abs_error_m_s=max(errors),main_integration_state_exactly_unchanged=True,
    whole_sole_clearance_matches_independent_support_function=True,
    main_FK_cache_stale_intentionally=True,
    evaluator_sha256=hashlib.sha256((SOURCE/'clearance_evaluation.py').read_bytes()).hexdigest())
(HERE/'eval-geometry-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

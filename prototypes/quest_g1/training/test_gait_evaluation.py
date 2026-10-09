import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from gait_evaluation import RAW_COLUMNS,runtime_env_class,summarize_gait,yaw_wxyz


def fixture(direction=(1.,0.)):
    t=np.arange(400)*.01
    a=np.zeros((len(t),len(RAW_COLUMNS)))
    def setcol(name,x):a[:,RAW_COLUMNS.index('gait_'+name)]=x
    setcol('ref_x',t);setcol('robot_x',direction[0]*t);setcol('robot_y',direction[1]*t)
    return a,setcol


class GaitTests(unittest.TestCase):
    def test_direction_is_not_path_length(self):
        for direction,expected,lateral in [((1,0),1,0),((0,1),0,1),((-1,0),-1,0)]:
            a,_=fixture(direction);m=summarize_gait(a,RAW_COLUMNS)
            self.assertAlmostEqual(m['path_length_ratio'],1.)
            self.assertAlmostEqual(m['directed_progress_ratio'],expected)
            self.assertEqual(m['lateral_error_abs_mean_m']>0,bool(lateral))

    def test_full_quaternion_heading_with_tilt(self):
        for r,p,y in [(.7,.5,1.2),(-.5,.3,-2.8),(0,0,3.13)]:
            q=Rotation.from_euler('xyz',[r,p,y]).as_quat()[[3,0,1,2]]
            self.assertAlmostEqual(yaw_wxyz(q*1.1),y)

    def test_fixed_reference_trip_and_swing(self):
        a,setcol=fixture();h=np.zeros(400);h[150:180]=.02
        v=np.zeros(400);v[150:180]=.5
        f=np.zeros(400);f[160:]=100
        setcol('ref_height_l',h);setcol('ref_speed_l',v);setcol('height_l',.005)
        setcol('force_l',f);setcol('brake_l',60)
        m=summarize_gait(a,RAW_COLUMNS)
        self.assertEqual(m['trips'],1);self.assertEqual(m['braking_trips'],1)
        self.assertEqual(m['complete_swings'],1)
        self.assertAlmostEqual(m['swing_peak_median_m'],.005)
        self.assertEqual(m['swings_below_1cm_fraction'],1)
        # A paired prefix ending mid-swing must not invent a complete swing.
        short=summarize_gait(a[:170],RAW_COLUMNS)
        self.assertEqual(short['complete_swings'],0)
        self.assertEqual(short['trips'],1)

    def test_stationary_reference_has_no_progress_ratio(self):
        a,setcol=fixture();setcol('ref_x',0)
        self.assertIsNone(summarize_gait(a,RAW_COLUMNS)['directed_progress_ratio'])

    def test_history_cadence_and_action_hold(self):
        class Base:
            def __init__(self):self.steps=0;self.history=[];self.actions=[];self.last_obs=None
            def observation(self):
                self.history.append(self.steps);self.last_obs=self.steps;return self.last_obs
            def reset(self):return self.observation()
            def step(self,action):
                self.actions.append(action);self.steps+=1;return self.observation(),0,False,{}
        for hz in (50,100):
            e=runtime_env_class(Base,hz)();obs=e.reset();calls=[]
            for _ in range(20):
                if e.steps%e.policy_stride==0:action=obs;calls.append(obs)
                obs,*_=e.step(action)
            self.assertEqual(e.history,list(range(0,21,100//hz)))
            self.assertEqual(calls,list(range(0,20,100//hz)))
            if hz==50:self.assertEqual(e.actions,[i//2*2 for i in range(20)])

    def test_paired_prefix_recomputes_gait(self):
        from evaluate import compare, METRICS
        raw,_=fixture();columns=METRICS+RAW_COLUMNS
        trace=np.c_[np.zeros((len(raw),len(METRICS))),raw]
        trial={'trace_key':'clip','stage':2,'fell':False}
        b={'dataset_manifest_sha256':'same','trace_columns':columns,'control_hz':50,
           'gait_schema':'fixed-qpos-reference-v1','trials':[trial],'falls':0,
           'survived_seconds':4.,'_traces':{'clip':trace}}
        c={**b,'_traces':{'clip':trace[:250]}}
        result=compare(b,c)
        pair=result['paired_gait'][0]
        self.assertEqual(pair['prefix_seconds'],2.5)
        self.assertEqual(pair['baseline'],pair['candidate'])
        self.assertAlmostEqual(pair['baseline']['seconds'],1.5)

    def test_frequency_mismatch_rejected(self):
        from evaluate import compare
        b={'dataset_manifest_sha256':'same','trace_columns':[], 'control_hz':50}
        c={**b,'control_hz':100}
        with self.assertRaisesRegex(ValueError,'control_hz'):compare(b,c)


if __name__=='__main__':unittest.main()

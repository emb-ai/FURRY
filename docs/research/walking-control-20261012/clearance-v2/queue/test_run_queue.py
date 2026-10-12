"""CPU-only queue tests. Never run GPU commands, SSH, or a training process."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_queue as q


def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def gpu(index=0,memory=0,util=0,pids=None):return dict(index=index,uuid='GPU-'+str(index),memory_mib=memory,util_percent=util,compute_pids=pids or [])
def spec():
    s=dict(label=q.SMOKE,gpu=0,argv=['python','/code/clearance_finetune.py','--root','/root','--output','/code/'+q.SMOKE],
           worker_argv=['python','/queue/run_queue.py','worker'],code='/code',root='/root',training_output='/code/'+q.SMOKE,scheduled_epoch=1)
    s['identity_sha256']=q.seal_spec(s);return s
def process(pid,argv,birth=10):return dict(pid=pid,start_ticks=birth,argv=argv)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name).resolve()
        self.root=self.base/'root';self.code=self.base/'code';self.state=self.base/'state'
        for p in (self.root,self.code,self.state):p.mkdir()
        self.args=argparse.Namespace(root=self.root,code=self.code,state=self.state,poll=15,gpus=list(range(8)),once=True)
    def tearDown(self):self.tmp.cleanup()

    def test_exact_idle_gpu_and_all_eight_devices(self):
        self.assertEqual(q.select_gpu([gpu(0,101),gpu(1,99,1),gpu(2,pids=[333]),gpu(7,99)],range(8),set(),0),7)
        self.assertIsNone(q.select_gpu([gpu(0,100)],range(8),set(),0))
        self.assertIsNone(q.select_gpu([gpu(0,99,0,[99])],range(8),set(),0))
    def test_reserved_gpu_and_max_four(self):
        self.assertEqual(q.select_gpu([gpu(0),gpu(3)],range(8),{0},1),3)
        self.assertIsNone(q.select_gpu([gpu(0)],range(8),set(),4))
    def test_gpu_uuid_compute_mapping(self):
        rows=q.parse_gpu_snapshot('7, GPU-A, 12, 0\n2, GPU-B, 0, 0\n','GPU-A, 42\n')
        self.assertEqual(rows[0]['compute_pids'],[42]);self.assertEqual(q.select_gpu(rows,range(8),set(),0),2)
        with self.assertRaises(RuntimeError):q.parse_gpu_snapshot('0, GPU-A, 0, 0','GPU-X, 2')
    def test_exact_birth_identity_and_pid_reuse(self):
        s=spec();a=process(30,s['argv']);b=process(30,s['argv'],11)
        self.assertTrue(q.exact_identity(a,a));self.assertFalse(q.exact_identity(a,b))
        with self.assertRaises(RuntimeError):q.job_status(s,dict(child=a),None,[b],now=100)
        self.assertEqual(q.job_status(s,dict(child=a),None,[a],now=100)['status'],'active')
    def test_pid_reuse_for_other_command_cannot_be_adopted(self):
        s=spec();a=process(30,s['argv']);different=process(30,['python','foreign.py'],11)
        self.assertEqual(q.job_status(s,dict(child=a),None,[different],now=100)['status'],'lost')
    def test_orphan_trainer_keeps_gpu_reserved_after_pipeline_exit(self):
        s=spec();s['argv']=['python','/code/clearance_pipeline.py'];s['identity_sha256']=q.seal_spec(s)
        trainer=process(60,['python','/code/clearance_finetune.py','--output',s['training_output'],'--root','/root'])
        done=dict(returncode=0,spec_sha256=s['identity_sha256'])
        self.assertEqual(q.job_status(s,None,done,[trainer],now=100)['status'],'active')
    def test_missing_receipt_never_retries_and_exit_identity_required(self):
        s=spec();self.assertEqual(q.job_status(s,None,None,[],now=100)['status'],'lost')
        self.assertEqual(q.job_status(s,None,None,[],now=10)['status'],'starting')
        with self.assertRaises(RuntimeError):q.job_status(s,None,dict(returncode=0,spec_sha256='wrong'),[],now=100)
        self.assertEqual(q.job_status(s,None,dict(returncode=1,spec_sha256=s['identity_sha256']),[],now=100)['status'],'failed')
    def test_bounded_immutable_plan(self):
        plans=q.plans(self.root,self.code);self.assertEqual(len(plans),5)
        self.assertEqual([p['label'] for p in plans[1:]],[r[0] for r in q.RUNS])
        for p in plans:
            self.assertIn('--window-reward',p['argv']);self.assertIn('--nonlooping',p['argv'])
            self.assertEqual(q.flag_value(p['argv'],'--init-checkpoint'),str(self.root/'comparison-baseline.pt'))
        for p in plans[1:]:
            self.assertEqual(q.flag_value(p['argv'],'--iterations'),'1000')
            self.assertEqual(q.flag_value(p['argv'],'--save-every'),'250')
            self.assertEqual(q.flag_value(p['argv'],'--num-envs'),'4096')
        self.assertEqual(sum('--velocity-clearance' in p['argv'] for p in plans[1:]),2)
    def test_duplicate_spec_refused_before_any_spawn(self):
        p=q.plans(self.root,self.code)[0];(self.state/(p['label']+'.spec.json')).write_text('{}')
        with patch.object(q.subprocess,'Popen') as popen:
            with self.assertRaises(RuntimeError):q.launch(p,0,[gpu()],self.args)
            popen.assert_not_called()
    def test_stop_preserves_jobs_and_never_launches(self):
        (self.state/'STOP').touch()
        with patch.object(q,'verify_inputs',return_value={}),patch.object(q,'process_snapshot',return_value=[]),patch.object(q,'gpu_snapshot') as snapshot,patch.object(q,'launch') as start:
            self.assertEqual(q.run(self.args),0);snapshot.assert_not_called();start.assert_not_called()
        status=q.read(self.state/'status.json');self.assertEqual(status['phase'],'stopped');self.assertEqual(len(status['jobs']),5)
    def test_busy_poll_does_not_rehash_dataset(self):
        with patch.object(q,'verify_inputs',return_value={}) as verify,patch.object(q,'process_snapshot',return_value=[]),patch.object(q,'gpu_snapshot',return_value=[gpu(0,4000)]),patch.object(q,'launch') as start:
            q.run(self.args);self.assertEqual(verify.call_count,1);start.assert_not_called()
    def test_gpu_rechecked_after_full_hashing(self):
        with patch.object(q,'verify_inputs',return_value={}) as verify,patch.object(q,'process_snapshot',return_value=[]),patch.object(q,'gpu_snapshot',side_effect=[[gpu()],[gpu()],[gpu(0,2000)]]),patch.object(q,'launch') as start:
            q.run(self.args);self.assertEqual(verify.call_count,2);start.assert_not_called()
    def test_one_idle_device_launches_smoke_only(self):
        with patch.object(q,'verify_inputs',return_value={}),patch.object(q,'process_snapshot',return_value=[]),patch.object(q,'gpu_snapshot',return_value=[gpu(7)]),patch.object(q,'launch') as start:
            q.run(self.args);self.assertEqual(start.call_count,1)
            self.assertEqual(start.call_args.args[0]['label'],q.SMOKE);self.assertEqual(start.call_args.args[1],7)
    def test_singleton_lock_blocks_second_queue(self):
        with (self.state/'queue.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with patch.object(q,'verify_inputs') as verify:
                with self.assertRaises(q.QueueBusy):q.run(self.args)
                verify.assert_not_called()

    def mock_smoke(self):
        folder=self.code/q.SMOKE;folder.mkdir()
        q.dump(folder/'config.json',dict(ppo_config_matches_upstream=True,
            velocity_clearance=dict(critic_inputs=1743,critic_features_added=0,enabled=True),
            initialization=dict(sha256=q.INIT_SHA),dataset_manifest_sha256=q.DATA_SHA,sole_urdf_sha256=q.SOLE_SHA))
        q.dump(folder/'gpu-contract-checks.json',dict(success=True,num_envs=4096,actor_inputs=1432))
        (folder/'metrics.jsonl').write_text(json.dumps(dict(phase='complete',actor_updates=1))+'\n');(folder/'final.pt').touch()
        return folder
    def test_smoke_gates_exit_actor_and_critic(self):
        folder=self.mock_smoke();self.assertTrue(q.verify_smoke(folder,dict(returncode=0)))
        with self.assertRaises(RuntimeError):q.verify_smoke(folder,dict(returncode=1))
        cfg=q.read(folder/'config.json');cfg['velocity_clearance']['critic_inputs']=1749;q.dump(folder/'config.json',cfg)
        with self.assertRaises(RuntimeError):q.verify_smoke(folder,dict(returncode=0))
    def test_matrix_waits_for_fresh_baseline(self):
        jobs={p['label']:dict(status='pending') for p in q.plans(self.root,self.code)};jobs[q.SMOKE]=dict(status='complete',gpu=0)
        q.dump(self.state/(q.SMOKE+'.exit.json'),dict(returncode=0))
        with patch.object(q,'verify_inputs',return_value={}),patch.object(q,'process_snapshot',return_value=[]),patch.object(q,'snapshot_jobs',return_value=jobs),patch.object(q,'verify_smoke'),patch.object(q,'baseline_ready',return_value=False),patch.object(q,'gpu_snapshot',return_value=[gpu()]),patch.object(q,'launch') as start:
            q.run(self.args);start.assert_not_called()
    def test_critic_not_ready_exit_zero_is_not_completed_training(self):
        label=q.RUNS[0][0];folder=self.root/'runs'/label/'training';folder.mkdir(parents=True)
        (folder/'metrics.jsonl').write_text(json.dumps(dict(phase='critic_not_ready',actor_updates=0))+'\n')
        with self.assertRaisesRegex(RuntimeError,'without training1000'):q.verify_training_complete(self.root,label)
    def test_training_needs_all_four_evaluations(self):
        label=q.RUNS[0][0];folder=self.root/'runs'/label;training=folder/'training';training.mkdir(parents=True)
        (training/'metrics.jsonl').write_text(json.dumps(dict(phase='complete',actor_updates=1000))+'\n');(training/'final.pt').touch()
        q.dump(folder/'result.json',dict(phase='pipeline_complete',results=[]))
        with self.assertRaises(RuntimeError):q.verify_training_complete(self.root,label)
        rows=[]
        for n in (250,500,750,1000):
            stem='checkpoint_%06d'%n;rows.append(dict(checkpoint=stem+'.pt'));(training/(stem+'.pt')).touch()
            for suffix in ('.json','.npz','.comparison.json','.clearance-verdict.json','.fingerprint.json'):(folder/(stem+suffix)).touch()
            q.dump(folder/(stem+'.json'),dict(total_trials=24))
        q.dump(folder/'result.json',dict(phase='pipeline_complete',results=rows))
        self.assertEqual(q.verify_training_complete(self.root,label)['actor_updates'],1000)
    def test_actual_dataset_bytes_are_verified(self):
        files={'comparison-baseline.pt':'actor','assets/g1_main_spheres.urdf':'sole','locomotion-ancestor-manifest.json':'ancestor','TWIST2/rsl_rl/rsl_rl/modules/actor_critic_future.py':'source','dataset/clip.pkl':'motion'}
        for name,text in files.items():
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
        q.dump(self.root/'dataset/manifest.json',dict(clips=[dict(id='clip',file='clip.pkl',sha256=sha('motion'))]))
        (self.code/'source.py').write_text('frozen');q.dump(self.code/'transfer-manifest.json',dict(files={'source.py':sha('frozen')}))
        with ExitStack() as stack:
            for key,value in [('SOURCE_SHA',q.digest(self.code/'transfer-manifest.json')),('INIT_SHA',sha('actor')),('DATA_SHA',q.digest(self.root/'dataset/manifest.json')),('SOLE_SHA',sha('sole'))]:stack.enter_context(patch.object(q,key,value))
            q.verify_inputs(self.root,self.code,True)
            (self.root/'dataset/clip.pkl').write_text('tampered')
            with self.assertRaisesRegex(RuntimeError,'Dataset clip changed'):q.verify_inputs(self.root,self.code,True)

    def deferred_fixture(self):
        plan=q.plans(self.root,self.code)[0]
        s=dict(plan,root=str(self.root),code=str(self.code),gpu=7,scheduled_epoch=1,worker_argv=['python','queue.py','worker'])
        s['identity_sha256']=q.seal_spec(s);q.dump(self.state/(q.SMOKE+'.spec.json'),s)
        q.dump(self.state/'queue-config.json',dict(inputs={}))
        self.args.label=q.SMOKE
        with patch.object(q,'verify_inputs',return_value={}),patch.dict(q.os.environ,{'CUDA_VISIBLE_DEVICES':'7'}),patch.object(q,'gpu_snapshot',return_value=[gpu(7,3000)]),patch.object(q.subprocess,'Popen') as popen:
            self.assertEqual(q.worker(self.args),75);popen.assert_not_called()
        return plan,s,q.read(self.state/(q.SMOKE+'.exit.json'))
    def test_documented_busy_before_child_can_be_requeued(self):
        plan,s,receipt=self.deferred_fixture()
        self.assertIs(receipt['child_started'],False)
        wrapper=process(100,s['worker_argv'])
        self.assertEqual(q.job_status(s,None,receipt,[wrapper],now=100)['status'],'active')
        with self.assertRaises(RuntimeError):q.requeue_deferred(plan,self.state,[wrapper])
        self.assertEqual(q.job_status(s,None,receipt,[],now=100)['status'],'deferred')
        q.requeue_deferred(plan,self.state,[])
        self.assertFalse((self.state/(q.SMOKE+'.spec.json')).exists())
        archive=self.state/'attempts'/q.SMOKE/s['identity_sha256']
        self.assertTrue((archive/(q.SMOKE+'.spec.json')).exists());self.assertTrue((archive/'retry-proof.json').exists())
    def test_deferred_cleanup_crash_recovered_without_ambiguous_retry(self):
        plan,s,receipt=self.deferred_fixture();archive=self.state/'attempts'/q.SMOKE/s['identity_sha256'];archive.mkdir(parents=True)
        q.dump(archive/(q.SMOKE+'.exit.json'),receipt);q.dump(archive/'retry-proof.json',dict(spec_sha256=s['identity_sha256']))
        (self.state/(q.SMOKE+'.exit.json')).unlink()
        status=q.snapshot_jobs(q.plans(self.root,self.code),self.state,[])
        self.assertEqual(status[q.SMOKE]['status'],'deferred')
        q.requeue_deferred(plan,self.state,[]);self.assertFalse((self.state/(q.SMOKE+'.spec.json')).exists())
    def test_no_retry_when_child_or_output_evidence_exists(self):
        plan,s,receipt=self.deferred_fixture();Path(plan['training_output']).mkdir()
        with self.assertRaises(RuntimeError):q.requeue_deferred(plan,self.state,[])
        receipt['child_started']=True
        self.assertEqual(q.job_status(s,None,receipt,[],now=100)['status'],'failed')


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(QueueTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(tests=result.testsRun,success=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),
                no_remote_actions=True,no_GPU_or_training_processes=True,queue_sha256=q.digest(Path(q.__file__)))
    Path(__file__).with_name('cpu-tests.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())

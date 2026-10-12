"""Independent stdlib queue checks. No GPU, SSH, or real process launches."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('reviewed_queue', HERE / 'run_queue.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class QueueReview(unittest.TestCase):
    def test_gpu_all_idle_conditions_and_cap(self):
        rows = queue.parse_gpu_snapshot(
            '0, GPU-a, 99, 0\n1, GPU-b, 100, 0\n2, GPU-c, 1, 2\n3, GPU-d, 1, 0',
            'GPU-d, 1234')
        self.assertEqual(queue.select_gpu(rows, [0, 1, 2, 3], set(), 0), 0)
        self.assertIsNone(queue.select_gpu(rows, [1, 2, 3], set(), 0))
        self.assertIsNone(queue.select_gpu(rows, [0, 1, 2, 3], {0}, 0))
        self.assertIsNone(queue.select_gpu(rows, [0], set(), 4))
        with self.assertRaises(RuntimeError):
            queue.parse_gpu_snapshot('0, GPU-a, 1, 0', 'GPU-unknown, 999')

    def test_fixed_matrix(self):
        plans = queue.plans(Path('/experiment'), Path('/source'))
        self.assertEqual(len(plans), 5)
        self.assertEqual(len({p['label'] for p in plans}), 5)
        self.assertIn('--smoke', plans[0]['argv'])
        self.assertEqual([p['seed'] for p in plans[1:]], [11, 22, 11, 22])
        for index, plan in enumerate(plans[1:]):
            argv = plan['argv']
            self.assertIn('--window-reward', argv)
            self.assertIn('--nonlooping', argv)
            self.assertNotIn('--resume', argv)
            self.assertEqual(queue.flag_value(argv, '--num-envs'), '4096')
            self.assertEqual(queue.flag_value(argv, '--iterations'), '1000')
            self.assertEqual(queue.flag_value(argv, '--save-every'), '250')
            self.assertEqual(queue.flag_value(argv, '--control-hz'), '50')
            self.assertEqual('--velocity-clearance' in argv, index >= 2)

    def test_birth_identity_not_pid_alone(self):
        original = dict(pid=1, start_ticks=10, argv=['python', 'ours'])
        self.assertTrue(queue.exact_identity(original, dict(original)))
        self.assertFalse(queue.exact_identity(original, dict(original, start_ticks=11)))
        self.assertFalse(queue.exact_identity(original, dict(original, argv=['foreign'])))

    def test_seal_rejects_mutated_argv(self):
        spec = dict(label='job', argv=['python', 'ours'], gpu=0)
        spec['identity_sha256'] = queue.seal_spec(spec)
        self.assertEqual(spec['identity_sha256'], queue.seal_spec(spec))
        spec['argv'].append('--different')
        self.assertNotEqual(spec['identity_sha256'], queue.seal_spec(spec))

    def scenario(self, smoke='pending', baseline=False, active=0, busy=False,
                 stop=False, failure=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'root'; root.mkdir()
            code = Path(temporary) / 'code'; code.mkdir()
            state = Path(temporary) / 'state'; state.mkdir()
            args = SimpleNamespace(root=root, code=code, state=state, gpus=list(range(8)), poll=15, once=True)
            plans = queue.plans(root, code)
            jobs = {p['label']: dict(status='pending') for p in plans}
            jobs[queue.SMOKE] = dict(status=smoke, gpu=7)
            for i in range(active):
                jobs[plans[i+1]['label']] = dict(status='active', gpu=i)
            if failure:
                jobs[plans[1]['label']] = dict(status=failure, gpu=0)
            if stop:
                (state/'STOP').touch()
            if smoke == 'complete':
                (state/(queue.SMOKE+'.exit.json')).write_text('{"returncode": 0}')
            rows = [dict(index=i, uuid='GPU-'+str(i), memory_mib=1,
                         util_percent=0, compute_pids=[]) for i in range(8)]
            current = [dict(row, compute_pids=[9000+i]) if busy else row for i, row in enumerate(rows)]
            launches = []
            with patch.object(queue, 'verify_inputs', return_value={'pinned':'yes'}), \
                 patch.object(queue, 'process_snapshot', return_value=[]), \
                 patch.object(queue, 'snapshot_jobs', return_value=jobs), \
                 patch.object(queue, 'verify_smoke', return_value={'success':True}) as smoke_check, \
                 patch.object(queue, 'baseline_ready', return_value=baseline) as baseline_check, \
                 patch.object(queue, 'gpu_snapshot', side_effect=[rows,current,current]), \
                 patch.object(queue, 'launch', side_effect=lambda plan,*rest: launches.append(plan['label'])):
                if failure:
                    with self.assertRaises(RuntimeError): queue.run(args)
                else:
                    queue.run(args)
                return launches, smoke_check.call_count, baseline_check.call_count

    def test_smoke_before_any_training(self):
        self.assertEqual(self.scenario()[0], [queue.SMOKE])
        self.assertEqual(self.scenario(smoke='active')[0], [])

    def test_baseline_and_smoke_both_required(self):
        self.assertEqual(self.scenario(smoke='complete', baseline=False), ([],1,1))
        launches, smoke_calls, baseline_calls = self.scenario(smoke='complete', baseline=True)
        self.assertEqual(launches, [queue.RUNS[0][0]])
        self.assertEqual((smoke_calls,baseline_calls), (1,1))

    def test_max_four_and_fresh_gpu_query(self):
        self.assertEqual(self.scenario(smoke='complete', baseline=True, active=4)[0], [])
        self.assertEqual(self.scenario(smoke='complete', baseline=True, busy=True)[0], [])

    def test_stop_failure_and_lost_prevent_new_jobs(self):
        self.assertEqual(self.scenario(stop=True)[0], [])
        self.assertEqual(self.scenario(smoke='complete', baseline=True, failure='failed')[0], [])
        self.assertEqual(self.scenario(smoke='complete', baseline=True, failure='lost')[0], [])

    def test_exit_zero_does_not_hide_unready_critic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder=root/'runs'/'trial';training=folder/'training'
            training.mkdir(parents=True)
            (training/'metrics.jsonl').write_text('{"phase":"critic_not_ready","actor_updates":0}\n')
            with self.assertRaisesRegex(RuntimeError,'without training1000 completion'):
                queue.verify_training_complete(root,'trial')
            (training/'metrics.jsonl').write_text('{"phase":"complete","actor_updates":1000}\n')
            (training/'final.pt').write_bytes(b'test final')
            names=['checkpoint_%06d.pt'%n for n in (250,500,750,1000)]
            (folder/'result.json').write_text(json.dumps(dict(phase='pipeline_complete',results=[dict(checkpoint=n,accepted=False) for n in names])))
            for name in names:
                (training/name).write_bytes(b'test checkpoint')
                for suffix in ('.json','.npz','.comparison.json','.clearance-verdict.json','.fingerprint.json'):
                    (folder/(Path(name).stem+suffix)).write_text('{"total_trials":24}')
            result=queue.verify_training_complete(root,'trial')
            self.assertTrue(result['training_complete'])
            self.assertEqual(result['actor_updates'],1000)
            # Training completion does not imply any candidate was accepted.
            self.assertNotIn('accepted',result)
            (folder/'checkpoint_001000.npz').unlink()
            with self.assertRaisesRegex(RuntimeError,'Missing evaluation artifact'):
                queue.verify_training_complete(root,'trial')

    def test_retry_only_proven_no_child_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary).resolve();root=base/'root';code=base/'code';state=base/'state'
            for directory in (root,code,state):directory.mkdir()
            plan=queue.plans(root,code)[0]
            spec=dict(plan,root=str(root),code=str(code),gpu=0,scheduled_epoch=1,
                      worker_argv=['python','queue.py','worker'])
            spec['identity_sha256']=queue.seal_spec(spec)
            queue.dump(state/(plan['label']+'.spec.json'),spec)
            queue.dump(state/'queue-config.json',dict(inputs={}))
            args=SimpleNamespace(root=root,code=code,state=state,label=plan['label'])
            busy=[dict(index=0,uuid='GPU-0',memory_mib=9000,util_percent=100,compute_pids=[100])]
            with patch.object(queue,'verify_inputs',return_value={}), \
                 patch.dict(queue.os.environ,{'CUDA_VISIBLE_DEVICES':'0'}), \
                 patch.object(queue,'gpu_snapshot',return_value=busy), \
                 patch.object(queue.subprocess,'Popen') as popen:
                self.assertEqual(queue.worker(args),75)
                popen.assert_not_called()
            receipt=queue.read(state/(plan['label']+'.exit.json'))
            self.assertIs(receipt['child_started'],False)
            self.assertEqual(receipt['spec_sha256'],spec['identity_sha256'])
            live=dict(pid=99,start_ticks=123,argv=spec['worker_argv'])
            with self.assertRaises(RuntimeError):queue.requeue_deferred(plan,state,[live])
            self.assertEqual(queue.job_status(spec,None,None,[],now=100)['status'],'lost')
            self.assertEqual(queue.job_status(spec,None,dict(returncode=75,spec_sha256=spec['identity_sha256']),[],now=100)['status'],'failed')
            queue.requeue_deferred(plan,state,[])
            self.assertFalse((state/(plan['label']+'.spec.json')).exists())
            archive=state/'attempts'/plan['label']/spec['identity_sha256']
            self.assertEqual(queue.read(archive/(plan['label']+'.exit.json')),receipt)
            self.assertTrue((archive/'retry-proof.json').exists())


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(QueueReview))
    report = dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests_run=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors),
                  queue_sha256=hashlib.sha256((HERE/'run_queue.py').read_bytes()).hexdigest(),
                  scope='Independent local orchestration checks only; no remote execution or real launches')
    (HERE/'independent-review-tests.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)

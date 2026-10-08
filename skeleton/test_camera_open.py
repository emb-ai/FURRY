import unittest
from types import SimpleNamespace
from camera_open import open_pipeline

class OpenTest(unittest.TestCase):
    def test_failed_start_releases_stream(self):
        class Pipeline:
            stopped=False
            def start(self,c):raise RuntimeError('failed power state')
            def stop(self):self.stopped=True
        p=Pipeline()
        with self.assertRaises(RuntimeError):open_pipeline(SimpleNamespace(pipeline=lambda:p),None)
        self.assertTrue(p.stopped)

    def test_optional_clock_failure_preserves_stream(self):
        class Sensor:
            def supports(self,x):return True
            def set_option(self,*args):raise RuntimeError('clock option')
        profile=SimpleNamespace(get_device=lambda:SimpleNamespace(query_sensors=lambda:[Sensor()]))
        p=SimpleNamespace(start=lambda _:profile)
        module=SimpleNamespace(pipeline=lambda:p,option=SimpleNamespace(global_time_enabled=1))
        self.assertEqual(open_pipeline(module,None),(p,profile))

if __name__=='__main__':unittest.main()

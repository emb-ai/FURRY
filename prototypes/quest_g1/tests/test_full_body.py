from pathlib import Path
import os
import sys
import subprocess
import tempfile
import unittest
import struct

class FullBodyTests(unittest.TestCase):
    def test_native_roundtrip_and_episode_lifecycle(self):
        import mujoco
        root = Path(__file__).resolve().parents[1]
        runtime = Path(mujoco.__file__).parent
        library = next(runtime.glob('libmujoco.*'))
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory)/'check_full_body'
            subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror',
                '-I'+str(root/'android/native'),'-I'+str(runtime/'include'),
                str(root/'tests/check_full_body.cpp'),str(library),'-o',str(binary)],check=True)
            env=dict(os.environ);env['DYLD_LIBRARY_PATH' if sys.platform=='darwin' else 'LD_LIBRARY_PATH']=str(runtime)
            subprocess.run([str(binary),directory],env=env,check=True)
            raw=(Path(directory)/'body_full.bin').read_bytes()
            self.assertEqual(struct.unpack_from('<8sIII',raw), (b'G1BODY84',1,3072,84))
            header=struct.unpack_from('<QqqqIfBBBBI',raw,20)
            self.assertEqual(header,(42,987654321012345,123456789012345,123456789012245,7,.875,1,1,1,1,84))
            self.assertEqual(struct.unpack_from('<Q',raw,20+3072+48+3*36)[0],1<<60)
            sys.path.insert(0,str(root/'scripts'))
            from inspect_full_body import inspect
            report=inspect(directory,Path(directory)/'raw.csv')
            self.assertEqual((report['frames'],report['active'],report['skeletons']),(3,1,1))
            self.assertEqual(report['valid_position_orientation_frames'],[1]*84)
            (Path(directory)/'body_full.bin').write_bytes(raw[:-1])
            with self.assertRaisesRegex(ValueError,'Truncated'):
                inspect(directory)


    def test_sdk_capture(self):
        import mujoco
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            tmp=Path(directory)
            (tmp/'android').mkdir()
            (tmp/'android/log.h').write_text('#pragma once\n#define ANDROID_LOG_WARN 1\n#define ANDROID_LOG_INFO 2\ninline int __android_log_print(int,const char*,const char*,...){return 0;}\n')
            binary=tmp/'check_body_tracking'
            subprocess.run(['c++','-std=c++17','-I'+str(tmp),'-I'+str(root/'android/native'),
                '-I'+str(root/'vendor/openxr-android/prefab/modules/headers/include'),'-I'+str(Path(mujoco.__file__).parent/'include'),
                str(root/'tests/check_body_tracking.cpp'),'-o',str(binary)],check=True)
            subprocess.run([str(binary)],check=True)

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from rgbd_recording import RGBDRecorder

class RecordingTest(unittest.TestCase):
    def test_roundtrip_and_input_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'capture';r=RGBDRecorder(out)
            color=np.ones((8,10,3),np.uint8);depth=np.ones((8,10),np.float32)*2
            r.submit(color,depth,[100,101,5,4],123456,7)
            color[:]=0;depth[:]=0;r.close()
            with np.load(out/'frame-000000007.npz') as f:
                self.assertTrue(np.all(f['color']==1));self.assertTrue(np.all(f['depth_m']==2))
                self.assertEqual(int(f['epoch_ms']),123456);self.assertEqual(int(f['sequence']),7)
            self.assertEqual(json.loads((out/'manifest.json').read_text())['frames'],1)
            with self.assertRaises(FileExistsError):RGBDRecorder(out)

if __name__=='__main__':unittest.main()

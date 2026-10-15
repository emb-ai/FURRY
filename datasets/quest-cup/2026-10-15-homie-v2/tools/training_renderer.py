"""Same-frame robot state + previous sampled gaze, no shadow maps.

Capture writes body reference before integration but turns the saved camera with
current operator gaze. At inference that current gaze is not yet predicted.
Use last sample's yaw/pitch on the current physical ego-camera transform.
The first frame uses its archived gaze as an explicit episode initial state.
"""
import numpy as np
from PIL import Image
from g1_sim.visual_dataset import EpisodeRenderer

class TrainingRenderer(EpisodeRenderer):
 def render_observation(self,index,previous_index):
  # Populate physical qpos/qvel, mocap, userdata, and recorded camera origin.
  super().render(index)
  c=self.episode['camera']
  yaw=float(c['gaze_yaw'][previous_index]);pitch=float(c['gaze_pitch'][previous_index])
  cy,sy=np.cos(yaw),np.sin(yaw);cp,sp=np.cos(pitch),np.sin(pitch)
  ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
  rx=np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]])
  rotation=self.data.cam_xmat[self.camera_id].reshape(3,3)@ry@rx
  for cam in self.renderer.scene.camera:
   cam.forward[:]=-rotation[:,2];cam.up[:]=rotation[:,1]
  self.renderer.scene.flags[self.mj.mjtRndFlag.mjRND_SHADOW]=0
  return np.asarray(Image.fromarray(self.renderer.render()).resize((224,224),Image.Resampling.BILINEAR))

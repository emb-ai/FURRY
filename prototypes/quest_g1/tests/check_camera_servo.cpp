#include "camera_pose_servo.h"
#include <stdexcept>
#include <cstdio>
int main(){
 std::array<float,35> ref{};TrackedPose actual,goal;goal.position={.5,0,0};
 // MuJoCo camera forward is +X, local -Z; yaw represented in world Z.
 double cameraR[9]={0,0,-1,-1,0,0,0,1,0};mju_mat2Quat(actual.quaternion.data(),cameraR);goal.quaternion=actual.quaternion;
 double root[4]={1,0,0,0};auto command=CameraPoseServo(ref,goal,actual,root);
 if(command[0]<=0 || std::abs(command[1])>1e-6 || command[0]>.4f)throw std::runtime_error("No bounded stopped-target correction");
 double q[4]={std::cos(.1),0,0,std::sin(.1)};mju_mulQuat(goal.quaternion.data(),q,actual.quaternion.data());command=CameraPoseServo(ref,goal,actual,root);
 if(std::abs(command[5]-.2)>1e-6)throw std::runtime_error("Heading goal not followed");
 command=CameraPoseServo(ref,goal,actual,root,root);
 if(std::abs(command[5]-.6)>1e-6)throw std::runtime_error("V2 heading correction missing");
 goal=actual;command=CameraPoseServo(ref,goal,actual,root);for(float v:command)if(v!=0)throw std::runtime_error("Residual command at goal");
 ref[0]=20;ref[1]=20;ref[5]=20;command=CameraPoseServo(ref,goal,actual,root);
 if(std::hypot(command[0],command[1])>.800001 || command[5]>1.2f)throw std::runtime_error("Unbounded velocity");
 {
  std::array<float,35> forward{};forward[0]=.2f;forward[12]=.37f;
  double rotatedRoot[4]={std::sqrt(.5),0,0,std::sqrt(.5)};
  auto corrected=CameraPoseServo(forward,actual,actual,rotatedRoot,root);
  if(std::abs(corrected[0])>1e-6||std::abs(corrected[1]+.2f)>1e-6||corrected[12]!=forward[12])
   throw std::runtime_error("Reference/world/actual velocity frames disagree");
  auto aligned=CameraPoseServo(forward,actual,actual,root,root);
  if(aligned!=forward)throw std::runtime_error("Aligned frames changed feedforward");
 }
 for(int i=0;i<35;i++)ref[i]=float(i)*.1f;
 if(ApplyCameraCatchUp(ref,goal,nullptr,nullptr,-1,false)!=ref)throw std::runtime_error("Disabled catch-up altered the command");
 const char xml[]="<mujoco><worldbody><body pos='0 0 .8'><freejoint/><geom type='sphere' size='.1'/><camera name='ego' pos='0 0 .1'/></body></worldbody></mujoco>";
 mjVFS vfs;mj_defaultVFS(&vfs);mj_addBufferVFS(&vfs,"servo.xml",xml,sizeof(xml));char error[1024];
 mjModel* model=mj_loadXML("servo.xml",&vfs,error,sizeof(error));mj_deleteVFS(&vfs);
 if(!model)throw std::runtime_error(error);mjData* data=mj_makeData(model);mj_forward(model,data);
 data->qpos[0]=.2; // deliberately stale camera buffer, as after an integration step
 command=ApplyCameraCatchUp(ref,goal,model,data,0,true);
 TrackedPose current;std::copy_n(data->cam_xpos,3,current.position.begin());mju_mat2Quat(current.quaternion.data(),data->cam_xmat);
 if(std::abs(current.position[0]-.2)>1e-9||command!=CameraPoseServo(ref,goal,current,data->qpos+3))throw std::runtime_error("Catch-up did not use current camera kinematics");
 mj_deleteData(data);mj_deleteModel(model);
 puts("Camera pose feedback direction, convergence, yaw and velocity limits passed");
}

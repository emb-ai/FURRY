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
 goal=actual;command=CameraPoseServo(ref,goal,actual,root);for(float v:command)if(v!=0)throw std::runtime_error("Residual command at goal");
 ref[0]=20;ref[1]=20;ref[5]=20;command=CameraPoseServo(ref,goal,actual,root);
 if(std::hypot(command[0],command[1])>.800001 || command[5]>1.2f)throw std::runtime_error("Unbounded velocity");
 puts("Camera pose feedback direction, convergence, yaw and velocity limits passed");
}

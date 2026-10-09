#pragma once
#include "retarget.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
// A bounded outer pose loop around TWIST2's velocity-conditioned controller.
// Goal/actual are MuJoCo camera frames. Physical state is never overwritten.
inline std::array<float,35> CameraPoseServo(std::array<float,35> reference,
        const TrackedPose& goal,const TrackedPose& actual,const double* rootQuaternion){
 double R[9],error[3]={goal.position[0]-actual.position[0],goal.position[1]-actual.position[1],0},local[3];
 mju_quat2Mat(R,rootQuaternion);mju_mulMatTVec(local,R,error,3,3);
 double speed=std::hypot(local[0],local[1]),scale=speed>.4?.4/speed:1.;
 reference[0]+=float(scale*local[0]);reference[1]+=float(scale*local[1]);
 double commanded=std::hypot(reference[0],reference[1]);if(commanded>.8){reference[0]*=float(.8/commanded);reference[1]*=float(.8/commanded);}
 double targetR[9],actualR[9];mju_quat2Mat(targetR,goal.quaternion.data());mju_quat2Mat(actualR,actual.quaternion.data());
 double targetYaw=std::atan2(-targetR[5],-targetR[2]),actualYaw=std::atan2(-actualR[5],-actualR[2]);
 double yaw=std::atan2(std::sin(targetYaw-actualYaw),std::cos(targetYaw-actualYaw));
 reference[5]=std::clamp(reference[5]+float(std::clamp(yaw,-.8,.8)),-1.2f,1.2f);
 return reference;
}

inline std::array<float,35> ApplyCameraCatchUp(std::array<float,35> reference,
        const TrackedPose& goal,const mjModel* model,mjData* data,int camera,bool enabled){
 if(!enabled)return reference;
 if(camera<0)throw std::runtime_error("Catch-up requires the ego camera");
 // mj_step integrates qpos after updating camera buffers; refresh kinematics
 // so live control and replay use the same current physical pose.
 mj_kinematics(model,data);mj_comPos(model,data);mj_camlight(model,data);
 TrackedPose actual;std::copy_n(data->cam_xpos+3*camera,3,actual.position.begin());
 mju_mat2Quat(actual.quaternion.data(),data->cam_xmat+9*camera);
 return CameraPoseServo(reference,goal,actual,data->qpos+3);
}

#pragma once
#include "retarget.h"
#include <algorithm>
#include <cmath>
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

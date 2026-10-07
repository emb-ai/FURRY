#pragma once
#include "retarget.h"
#include <openxr/openxr.h>
#include <common/xr_linear.h>

// One cyclopean pose per XR frame. Each eye projects the same physical panel.
inline XrMatrix4x4f PoseMatrix(const TrackedPose& pose) {
    double rotation[9];mju_quat2Mat(rotation,pose.quaternion.data());
    XrMatrix4x4f matrix{};matrix.m[15]=1;
    for(int row=0;row<3;row++){
        for(int col=0;col<3;col++)matrix.m[4*col+row]=float(rotation[3*row+col]);
        matrix.m[12+row]=float(pose.position[row]);
    }
    return matrix;
}
inline XrMatrix4x4f EgoWorld(const XrMatrix4x4f& camera,const XrMatrix4x4f& anchor) {
    XrMatrix4x4f inverse,result;
    XrMatrix4x4f_InvertRigidBody(&inverse,&camera);
    XrMatrix4x4f_Multiply(&result,&anchor,&inverse);
    return result;
}

// Construct the calibration similarity transform once; callers keep it fixed.
// Only the eye view changes with later HMD movement.
inline XrMatrix4x4f CalibratedCameraWorld(const XrMatrix4x4f& camera,
                                    const XrMatrix4x4f& head,
                                    const XrMatrix4x4f& rotation,float scale){
    XrMatrix4x4f world{};world.m[15]=1;
    for(int r=0;r<3;r++){
        world.m[12+r]=head.m[12+r];
        for(int c=0;c<3;c++){
            world.m[4*c+r]=rotation.m[4*c+r]*scale;
            world.m[12+r]-=world.m[4*c+r]*camera.m[12+c];
        }
    }
    return world;
}
inline XrMatrix4x4f LevelEgoRotation(const XrMatrix4x4f& camera,const XrMatrix4x4f& head){
    double fx=-head.m[8],fz=-head.m[10],hn=std::hypot(fx,fz);
    if(hn<.2){fx=0;fz=-1;hn=1;}fx/=hn;fz/=hn;
    double yaw=std::atan2(-camera.m[9],-camera.m[8]);
    double stageToRobot[9]={fx,0,fz,fz,0,-fx,0,1,0},q[4]={std::cos(yaw/2),0,0,std::sin(yaw/2)},r[9],basis[9];
    mju_quat2Mat(r,q);mju_mulMatMat(basis,r,stageToRobot,3,3,3);
    XrMatrix4x4f inverse{};inverse.m[15]=1;
    for(int row=0;row<3;row++)for(int col=0;col<3;col++)inverse.m[4*col+row]=float(basis[3*col+row]);
    return inverse;
}

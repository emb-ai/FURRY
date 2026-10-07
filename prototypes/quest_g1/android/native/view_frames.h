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

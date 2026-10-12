#pragma once
#include <mujoco/mujoco.h>
#include <array>
#include <cmath>
#include <stdexcept>

// Offline prototype. A bind mismatch is not a physical lever on the marker.
// All arguments are in one robot/world frame; mappedHumanRelative already
// includes the source basis conversion and its anatomical scale.
// This changes only source targets, before grounding and before the IK solver.
class AnatomicalMarkerOffset {
public:
    using Vec = std::array<double,3>;
    using Quat = std::array<double,4>;
private:
    Vec morphologyInRoot_{}, rigidInMarker_{};
    bool calibrated_=false;
    static void Finite(const double* p,int n) {
        for(int k=0;k<n;k++)if(!std::isfinite(p[k]))
            throw std::runtime_error("Nonfinite marker calibration");
    }
    static Quat Unit(const Quat& q) {
        Finite(q.data(),4);auto result=q;
        if(mju_normalize4(result.data())<1e-12)
            throw std::runtime_error("Invalid marker quaternion");
        return result;
    }
    static Vec InverseRotate(const Vec& v,const Quat& q) {
        Quat inverse;Vec result;mju_negQuat(inverse.data(),q.data());
        mju_rotVecQuat(result.data(),v.data(),inverse.data());return result;
    }
public:
    void Calibrate(const Vec& rootPosition,const Quat& rootRotation,
                   const Vec& anatomicalPosition,const Vec& markerPosition,
                   const Quat& markerRotation,const Vec& mappedHumanRelative) {
        calibrated_=false;
        for(const auto* p:{&rootPosition,&anatomicalPosition,&markerPosition,&mappedHumanRelative})Finite(p->data(),3);
        auto root=Unit(rootRotation),marker=Unit(markerRotation);Vec morphology,rigid;
        for(int k=0;k<3;k++) {
            morphology[k]=anatomicalPosition[k]-rootPosition[k]-mappedHumanRelative[k];
            rigid[k]=markerPosition[k]-anatomicalPosition[k];
        }
        morphologyInRoot_=InverseRotate(morphology,root);
        rigidInMarker_=InverseRotate(rigid,marker);calibrated_=true;
    }
    // FK is required: body_pos alone is wrong for nonidentity fixed rotations
    // or an intervening rigid body. Reject a task attached across a moving DOF.
    void CalibrateFromModel(const mjModel* model,const mjData* bind,
                            int rootBody,int anatomicalBody,int markerBody,
                            const Vec& mappedHumanRelative) {
        calibrated_=false;
        if(!model||!bind)throw std::runtime_error("Missing bind model/data");
        for(int b:{rootBody,anatomicalBody,markerBody})
            if(b<=0||b>=model->nbody)throw std::runtime_error("Invalid marker body");
        int b=markerBody;
        while(b!=anatomicalBody) {
            if(b==0||model->body_jntnum[b]!=0)
                throw std::runtime_error("Task marker is not rigidly attached to anatomical body");
            b=model->body_parentid[b];
        }
        Vec root,anatomy,marker;Quat rootQ,markerQ;
        for(int k=0;k<3;k++) {
            root[k]=bind->xpos[3*rootBody+k];anatomy[k]=bind->xpos[3*anatomicalBody+k];
            marker[k]=bind->xpos[3*markerBody+k];
        }
        mju_mat2Quat(rootQ.data(),bind->xmat+9*rootBody);
        mju_mat2Quat(markerQ.data(),bind->xmat+9*markerBody);
        Calibrate(root,rootQ,anatomy,marker,markerQ,mappedHumanRelative);
    }
    Vec Target(const Vec& rootPosition,const Quat& rootRotation,
               const Vec& mappedHumanRelative,const Quat& markerRotation)const {
        if(!calibrated_)throw std::runtime_error("Uncalibrated marker offset");
        Finite(rootPosition.data(),3);Finite(mappedHumanRelative.data(),3);
        auto root=Unit(rootRotation),marker=Unit(markerRotation);Vec a,b,result;
        mju_rotVecQuat(a.data(),morphologyInRoot_.data(),root.data());
        mju_rotVecQuat(b.data(),rigidInMarker_.data(),marker.data());
        for(int k=0;k<3;k++)result[k]=rootPosition[k]+mappedHumanRelative[k]+a[k]+b[k];
        return result;
    }
    const Vec& MorphologyInRoot()const{return morphologyInRoot_;}
    const Vec& RigidInMarker()const{return rigidInMarker_;}
};

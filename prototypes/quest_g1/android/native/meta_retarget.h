#pragma once
#include "gmr.h"

// Source adapter, not a second IK solver. Meta bind-skeleton coordinates are
// aligned to the robot's anatomical T pose before entering the GMR task stack.
class MetaRetargeter {
    GmrRetargeter gmr;
    double basis[9]{},basisQuat[4]{};
    std::array<double,3> translationOrigin{},rootOrigin{};
    std::array<TrackedPose,14> offsets{},calibrationRest{};
    std::array<double,14> scales{};
    double rootScale=1,footHeight=0,cameraScale=1;
    TrackedPose cameraOrigin,cameraHeadOrigin;
    std::array<double,4> cameraRotationOffset{1,0,0,0};
    bool cameraTracking=false;
    uint32_t skeletonVersion=0;
    int64_t lastTime=0;
    std::array<double,36> lastQ{};
    std::array<float,35> mimic{};
public:
    bool calibrated=false;
    double error=0;
    explicit MetaRetargeter(const std::string& assets):gmr(assets){}
    void Calibrate(const mjModel* model,const mjData* data,const TrackingFrame& input);
    const std::array<float,35>& Solve(const TrackingFrame& input);
    void EnableCameraTracking(bool value){cameraTracking=value;if(!value)gmr.ClearCameraTarget();}
    double VisualScale()const{return 1/cameraScale;}
    const double* StageToRobotRotation()const{return basis;}
    const TrackedPose& CameraCalibrationPose()const{return cameraOrigin;}
    const TrackedPose& HeadCalibrationPose()const{return cameraHeadOrigin;}
    void Pause(){lastTime=0;}
    bool Compatible(const TrackingFrame& f)const;
    GmrRetargeter& solver(){return gmr;}
};

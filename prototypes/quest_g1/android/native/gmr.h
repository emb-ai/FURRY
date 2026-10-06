#pragma once
#include "retarget.h"
#include <string>
// Native port of GMR bb1bbe4 and Mink 0.0.13. See third_party licenses.
// Model, costs, two-stage solve, SE(3) residuals and joint limits match upstream.
struct GmrTask {
    std::string human,robot;
    int body=-1;
    double scale=1;
    std::array<double,4> costs{}; // position/orientation for each of two stages
    TrackedPose offset;
};
class GmrRetargeter {
    mjModel* model_=nullptr; mjData* data_=nullptr;
    std::vector<GmrTask> tasks_;
    double assumedHeight=1.8,ground=0;
    std::vector<TrackedPose> targets;
    double Error(int stage);
    void Step(int stage);
public:
    GmrRetargeter(const std::string& assets);
    ~GmrRetargeter();
    GmrRetargeter(const GmrRetargeter&)=delete;
    const mjModel* model()const{return model_;}
    mjData* data(){return data_;}
    const std::vector<GmrTask>& tasks()const{return tasks_;}
    void Reset(const double* qpos=nullptr);
    // Upstream xrobot input, in GMR world coordinates, quaternion wxyz.
    void SetHumanTargets(const std::vector<TrackedPose>& human,double height,bool offsetToGround);
    // Device adapter may supply already-scaled, anatomically aligned targets.
    void SetTargets(const std::vector<TrackedPose>& value);
    void Solve();
    double error=0;
    int iterations=0;
};

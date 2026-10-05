#pragma once
#include <mujoco/mujoco.h>
#include <array>
#include <vector>
#include <cstdint>

struct TrackedPose {
    std::array<double,3> position{};
    std::array<double,4> quaternion{1,0,0,0}; // w,x,y,z
};
struct TrackingFrame {
    TrackedPose head;
    std::array<TrackedPose,2> hands;
    bool valid=false;
    uint64_t sequence=0;
    int64_t xr_time_ns=0; // predicted display time in the OpenXR clock domain
    std::array<uint64_t,3> location_flags{}; // head, left, right: OpenXR space flags
    std::array<bool,2> hand_active{};
};

// Quest adaptation of TWIST2's tracked-pose -> robot reference -> policy path.
// This is bounded native wrist IK, not the upstream full-body GMR/PICO solver.
class ArmRetargeter {
    const mjModel* model;
    mjData* ik;
    std::array<int,29> joint{}, qadr{}, vadr{};
    std::array<int,2> wrists{};
    TrackingFrame origin;
    double basis[9]{};
    std::array<std::array<double,3>,2> initialPosition{};
    std::array<std::array<double,9>,2> initialRotation{};
    std::array<std::array<double,3>,2> filteredPosition{};
    std::vector<double> jacp,jacr;
    std::vector<int> bodyGroup;
public:
    explicit ArmRetargeter(const mjModel* model);
    ~ArmRetargeter();
    void Calibrate(const mjData* data,const TrackingFrame& input);
    std::array<float,29> Solve(const TrackingFrame& input);
    bool calibrated=false;
    double error_m=0;
    bool limited=false;
};

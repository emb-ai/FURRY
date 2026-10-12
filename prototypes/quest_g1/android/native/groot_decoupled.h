#pragma once
#include <algorithm>
#include <array>
#include <cmath>

// GR00T decoupled WBC / HOMIE v2 observation and motor contract, ported from
// NVlabs/GR00T-WholeBodyControl b042411fae38ee4d1af9aac82a37a1f8d14d6dd0.
// Source: Apache-2.0, Copyright 2026 NVIDIA CORPORATION & AFFILIATES.
// See policies/homie-v2/SOURCE_LICENSE.txt and provenance.json.
namespace grootdecoupled {
inline constexpr int Hz=50, SingleObservation=86, History=6, Observation=516, Actions=15;
inline constexpr std::array<double,29> Home={-.1,0,0,.3,-.2,0, -.1,0,0,.3,-.2,0, 0,0,0,
                                            0,0,0,0,0,0,0, 0,0,0,0,0,0,0};
inline constexpr std::array<double,29> Kp={150,150,150,200,40,40, 150,150,150,200,40,40,
                                          250,250,250, 100,100,40,40,20,20,20, 100,100,40,40,20,20,20};
inline constexpr std::array<double,29> Kd={2,2,2,4,2,2, 2,2,2,4,2,2,
                                          5,5,5, 5,5,2,2,2,2,2, 5,5,2,2,2,2,2};
struct Command {
    std::array<float,3> velocity{}; // pelvis-local vx, vy, yaw rate
    float height=.74f;
    std::array<float,3> torsoRpy{}; // torso orientation relative to yaw-only pelvis
    std::array<float,14> arms{}; // direct upper-body IK, absent from RL action
};
inline bool Walking(const Command& command) {
    double squared=0;for(float value:command.velocity)squared+=double(value)*value;
    return std::sqrt(squared)>=.05;
}
inline std::array<float,SingleObservation> Observe(const Command& command,
        const std::array<double,29>& q,const std::array<double,29>& dq,
        const std::array<double,3>& omega,const std::array<double,4>& quat,
        const std::array<float,Actions>& previousAction) {
    std::array<float,SingleObservation> out{};
    out[0]=command.velocity[0]*2;out[1]=command.velocity[1]*2;out[2]=command.velocity[2]*.5f;
    out[3]=command.height;
    for(int i=0;i<3;i++){out[4+i]=command.torsoRpy[i];out[7+i]=omega[i]*.5;}
    const double w=quat[0],x=quat[1],y=quat[2],z=quat[3];
    out[10]=2*(w*y-x*z);out[11]=-2*(w*x+y*z);out[12]=1-2*(w*w+z*z);
    for(int i=0;i<29;i++){out[13+i]=q[i]-Home[i];out[42+i]=dq[i]*.05;}
    std::copy(previousAction.begin(),previousAction.end(),out.begin()+71);
    return out;
}
inline void Append(std::array<float,Observation>& history,
                   const std::array<float,SingleObservation>& observation) {
    std::move(history.begin()+SingleObservation,history.end(),history.begin());
    std::copy(observation.begin(),observation.end(),history.end()-SingleObservation);
}
} // namespace grootdecoupled

#pragma once
#include "file_sha256.h"
#include <array>
#include <cmath>
#include <set>

namespace bodymotors {
// Exact 29-body-motor subset of the pinned HOMIE deployed teleop XML, including
// motor inertia and force limits. This is distinct from its sim2mujoco test XML.
inline constexpr const char* Id="homie_teleop_b042411f_body_motors_v1";
inline constexpr const char* File="body_motor_profile.txt";
inline constexpr const char* Sha256="131fa5aae01c557238d24e2af0a2682f14e45286784b2605b35c4284526703ca";
struct Motor {
    std::string name;
    double armature=0,damping=0,friction=0;
    std::array<double,2> range{},jointForce{},control{},actuatorForce{};
    bool jointLimited=false,jointForceLimited=false,controlLimited=false,actuatorForceLimited=false;
};
using Profile=std::array<Motor,29>;
inline Profile Load(const std::string& path, const char* expectedId=Id, const char* expectedSha=Sha256) {
    questpolicy::VerifySha256(path,expectedSha);
    std::ifstream input(path);std::string id;input>>id;
    if(id!=expectedId)throw std::runtime_error("Unknown body motor profile");
    Profile out;std::set<std::string> names;
    for(auto& motor:out){
        if(!(input>>motor.name>>motor.armature>>motor.damping>>motor.friction
             >>motor.range[0]>>motor.range[1]>>motor.jointForce[0]>>motor.jointForce[1]) ||
           !names.insert(motor.name).second)
            throw std::runtime_error("Invalid body motor profile");
        motor.control=motor.jointForce;
        motor.jointLimited=motor.jointForceLimited=motor.controlLimited=true;
    }
    input>>std::ws;if(!input.eof())throw std::runtime_error("Extra body motor profile data");
    return out;
}
} // namespace bodymotors

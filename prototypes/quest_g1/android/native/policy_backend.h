#pragma once
#include <string>
#include <stdexcept>

namespace questpolicy {
// A backend is an observation/action contract, not a label for interchangeable weights.
inline constexpr const char* Twist2 = "twist2";
inline constexpr const char* GrootDecoupled = "groot_decoupled_v1";
inline constexpr const char* Sonic = "sonic_vr3_v1";
inline constexpr const char* SonicMotorFile="sonic_motor_profile.txt";
inline constexpr const char* SonicMotorId="sonic_b042411f_deploy_motors";
inline constexpr const char* SonicMotorSha="45f4e3dde56f723620178376a361d352656372da06ee394994d78bb209e06769";
inline bool KnownBackend(const std::string& backend) {
    return backend == Twist2 || backend == GrootDecoupled || backend == Sonic;
}
struct RuntimePolicy {
    std::string backend = Twist2;
    std::string path, secondaryPath;
    std::string motorProfilePath{}, plannerPath{};
    bool asyncPlanner=false;
    void Validate() const {
        if(!KnownBackend(backend) || path.empty() ||
           ((backend != Twist2) != !secondaryPath.empty()) ||
           ((backend == Sonic) != !plannerPath.empty()) ||
           (backend==Twist2 && !motorProfilePath.empty()))
            throw std::runtime_error("Invalid policy backend/files");
    }
};
} // namespace questpolicy

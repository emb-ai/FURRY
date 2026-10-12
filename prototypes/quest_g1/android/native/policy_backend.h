#pragma once
#include <string>
#include <stdexcept>

namespace questpolicy {
// A backend is an observation/action contract, not a label for interchangeable weights.
inline constexpr const char* Twist2 = "twist2";
inline constexpr const char* GrootDecoupled = "groot_decoupled_v1";
inline bool KnownBackend(const std::string& backend) {
    return backend == Twist2 || backend == GrootDecoupled;
}
struct RuntimePolicy {
    std::string backend = Twist2;
    std::string path, secondaryPath;
    void Validate() const {
        if(!KnownBackend(backend) || path.empty() ||
           ((backend == GrootDecoupled) != !secondaryPath.empty()))
            throw std::runtime_error("Invalid policy backend/files");
    }
};
} // namespace questpolicy

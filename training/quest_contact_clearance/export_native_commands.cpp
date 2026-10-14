// Export the input of Simulation::Step, before application blend/catch-up.
// Historical 14-marker captures cannot reconstruct the full-foot mode.
#include "meta_retarget.h"
#include "simulation.h"
#include <fstream>
#include <iomanip>
#include <iostream>

bool Read(std::istream& in, TrackingFrame& f) {
    if (!(in >> f.sequence >> f.xr_time_ns >> f.body.time_ns >> f.body.skeleton_version)) return false;
    f.valid = f.body.valid = f.body.supported = true;
    f.body.confidence = 1;
    f.location_flags[0] = 15;
    for (double& v : f.head.position) in >> v;
    for (double& v : f.head.quaternion) in >> v;
    for (int j=0; j<14; j++) {
        f.body.flags[j] = 15;
        for (auto* p : {&f.body.joints[j], &f.body.rest[j]}) {
            for (double& v : p->position) in >> v;
            for (double& v : p->quaternion) in >> v;
        }
    }
    if (!in) throw std::runtime_error("Truncated source frame");
    return true;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "assets epoch.txt output.csv camera_tracking(0|1)\n";
        return 2;
    }
    try {
        if (std::string(argv[4]) != "0" && std::string(argv[4]) != "1")
            throw std::runtime_error("Invalid camera tracking flag");
        Simulation sim(argv[1], 0);
        MetaRetargeter meta(argv[1]);
        meta.EnableCameraTracking(std::string(argv[4]) == "1");
        // The native source in main has no full-foot mode: 14 markers only,
        // as the contract below records.
        std::ifstream src(argv[2]); std::ofstream out(argv[3]);
        if (!src || !out) throw std::runtime_error("Cannot open stream");
        TrackingFrame f;
        if (!Read(src, f)) throw std::runtime_error("Missing calibration frame");
        meta.Calibrate(sim.model, sim.data, f);
        if (!meta.calibrated) throw std::runtime_error("Calibration failed");
        const auto calibrationSequence = f.sequence;
        out << std::setprecision(17) << "sequence,time_ns";
        for (int k=0; k<35; k++) out << ",cmd_" << k;
        for (int k=0; k<36; k++) out << ",qref_" << k;
        out << '\n';
        size_t count=0;
        while (Read(src, f)) {
            auto cmd = meta.Solve(f);
            out << f.sequence << ',' << f.body.time_ns;
            for (auto v : cmd) {
                if (!std::isfinite(v)) throw std::runtime_error("Nonfinite command");
                out << ',' << v;
            }
            for (int k=0; k<36; k++) out << ',' << meta.solver().data()->qpos[k];
            out << '\n';
            count++;
        }
        std::cout << "exported_frames=" << count << '\n';
        std::ofstream receipt(std::string(argv[3])+".metadata.json");
        receipt << "{\"camera_tracking\":" << argv[4]
                << ",\"calibration_sequence\":" << calibrationSequence
                << ",\"frames\":" << count
                << ",\"full_feet\":false,\"catch_up\":false,\"startup_blend\":false}\n";
        return count ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}

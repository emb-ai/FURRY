#include "meta_retarget.h"
#include "simulation.h"
#include "ablation_metrics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Qpos = std::array<double, 36>;
using Command = std::array<float, 35>;
constexpr double kPi = 3.14159265358979323846;
constexpr double kHumanScale = 1.25;
constexpr double kControlDt = .01;
constexpr double kStartupSeconds = 2;

int Body(const mjModel* model, const char* name) {
    int id = mj_name2id(model, mjOBJ_BODY, name);
    if (id < 0) throw std::runtime_error(std::string("Missing body: ") + name);
    return id;
}

int Joint(const mjModel* model, const std::string& name) {
    int id = mj_name2id(model, mjOBJ_JOINT, name.c_str());
    if (id < 0) throw std::runtime_error("Missing joint: " + name);
    return id;
}

std::vector<Qpos> LoadClip(const std::string& path, double& fps) {
    std::ifstream file(path);
    if (!file || !(file >> fps) || !std::isfinite(fps) || fps <= 0)
        throw std::runtime_error("Motion clip requires a positive finite source FPS");
    std::vector<Qpos> clip;
    Qpos frame{};
    while (file >> frame[0]) {
        for (int k = 1; k < 36; k++) {
            if (!(file >> frame[k])) throw std::runtime_error("Incomplete motion qpos frame");
        }
        for (double value : frame) {
            if (!std::isfinite(value)) throw std::runtime_error("Nonfinite motion qpos");
        }
        clip.push_back(frame);
    }
    if (!file.eof()) throw std::runtime_error("Invalid motion clip data");
    if (clip.empty()) throw std::runtime_error("Empty motion clip");
    return clip;
}

Qpos PhysicalPose(const Simulation& sim, const mjModel* model) {
    Qpos pose{};
    std::copy_n(sim.data->qpos, 7, pose.begin());
    for (int j = 1; j < model->njnt; j++) {
        int source = Joint(sim.model, mj_id2name(model, mjOBJ_JOINT, j));
        pose[model->jnt_qposadr[j]] = sim.data->qpos[sim.model->jnt_qposadr[source]];
    }
    return pose;
}

void CheckGeneratedLimits(const mjModel* model, const Qpos& pose) {
    for (int j = 1; j < model->njnt; j++) {
        double value = pose[model->jnt_qposadr[j]];
        if (!std::isfinite(value) || (model->jnt_limited[j] &&
            (value < model->jnt_range[2*j] - 1e-10 || value > model->jnt_range[2*j+1] + 1e-10))) {
            throw std::runtime_error(std::string("Generated arms truth exceeds joint limit: ") +
                                     mj_id2name(model, mjOBJ_JOINT, j));
        }
    }
}

std::vector<Qpos> ArmClip(const Simulation& sim, const mjModel* model, double fps) {
    Qpos home = PhysicalPose(sim, model);
    std::vector<Qpos> clip;
    for (int n = 0; n < 1200; n++) {
        auto pose = home;
        double t = n / fps;
        double reach = .5 * (1 - std::cos(2*kPi*t/6));
        // Independent smooth robot joint trajectories exercise both arms,
        // including forward/lateral reach, elbow bends and wrist rotation.
        // Their truth is known robot FK, not measured human motion.
        pose[22] = -.65*reach;
        pose[23] = .2 + .35*std::sin(t*.8);
        pose[24] = .25*std::sin(t*.6);
        pose[25] = 1.2 + .45*std::sin(t*.7);
        pose[26] = .25*std::sin(t);
        pose[27] = .15*std::sin(t*.8);
        pose[28] = .2*std::sin(t*.6);
        pose[29] = -.45*reach;
        pose[30] = -.2 - .3*std::sin(t*.65);
        pose[31] = -.2*std::sin(t*.7);
        pose[32] = 1.2 + .5*std::sin(t*.6);
        pose[33] = -.2*std::sin(t*.9);
        pose[34] = .15*std::sin(t*.5);
        pose[35] = -.2*std::sin(t*.75);
        CheckGeneratedLimits(model, pose);
        clip.push_back(pose);
    }
    return clip;
}

void FillSyntheticMeta(TrackingFrame& input, GmrRetargeter& g, const mjData* data, bool rest) {
    auto* model = g.model();
    auto& poses = rest ? input.body.rest : input.body.joints;
    const double xr[9] = {0,-1,0, 0,0,1, -1,0,0};
    // Geometric robot -> synthetic Meta -> robot roundtrip fixture. Positions
    // use anatomical pivots, while orientations use the GMR task frames.
    // This does not model Meta's inference errors or establish human accuracy.
    for (int i = 0; i < 14; i++) {
        int body = g.tasks()[i].body;
        if (i == 6 || i == 7)
            body = Body(model, i == 6 ? "left_ankle_roll_link" : "right_ankle_roll_link");
        if (i == 8 || i == 9)
            body = Body(model, i == 8 ? "left_shoulder_pitch_link" : "right_shoulder_pitch_link");
        const double* p = data->xpos + 3*body;
        poses[i].position = {-kHumanScale*p[1], kHumanScale*p[2], -kHumanScale*p[0]};
        double rotation[9];
        mju_mulMatMat(rotation, xr, data->xmat + 9*g.tasks()[i].body, 3, 3, 3);
        mju_mat2Quat(poses[i].quaternion.data(), rotation);
        input.body.flags[i] = 15;
    }
    // head_link's frame origin is near the torso. The model's explicit head
    // landmark supplies the height/lever arm needed to test HMD anchoring.
    const double* head = data->xpos + 3*Body(model, "head_mocap");
    input.head.position = {-kHumanScale*head[1], kHumanScale*head[2], -kHumanScale*head[0]};
}

Command DirectCommand(const mjModel* model, const Qpos& last, const Qpos& pose, double fps) {
    Command command{};
    double velocity[35], rotation[9], local[3];
    mj_differentiatePos(model, velocity, 1/fps, last.data(), pose.data());
    mju_quat2Mat(rotation, pose.data() + 3);
    mju_mulMatTVec(local, rotation, velocity, 3, 3);
    command[0] = local[0];
    command[1] = local[1];
    command[2] = pose[2];
    double w = pose[3], x = pose[4], y = pose[5], z = pose[6];
    command[3] = std::atan2(2*(w*x+y*z), 1-2*(x*x+y*y));
    command[4] = std::asin(std::clamp(2*(w*y-z*x), -1., 1.));
    command[5] = velocity[5];
    for (int k = 0; k < 29; k++) command[k+6] = pose[k+7];
    return command;
}

double TiltDegrees(const mjData* data) {
    double y = data->qpos[4], z = data->qpos[5];
    return std::acos(std::clamp(1-2*(y*y+z*z), -1., 1.))*180/kPi;
}

void CheckPhysicsNumerics(const Simulation& sim, double previousTime) {
    // MuJoCo may recover from invalid state by resetting mjData internally.
    // Such a rollout is invalid even when the recovered robot has not fallen.
    for (auto warning : {mjWARN_BADQPOS, mjWARN_BADQVEL, mjWARN_BADQACC, mjWARN_BADCTRL}) {
        if (sim.data->warning[warning].number) {
            const char* name = warning == mjWARN_BADQPOS ? "BADQPOS" :
                               warning == mjWARN_BADQVEL ? "BADQVEL" :
                               warning == mjWARN_BADQACC ? "BADQACC" : "BADCTRL";
            throw std::runtime_error(std::string("Invalid physics rollout: MuJoCo ") + name +
                                     " at index " + std::to_string(sim.data->warning[warning].lastinfo));
        }
    }
    if (!std::isfinite(sim.data->time) ||
        std::abs(sim.data->time - previousTime - sim.model->opt.timestep) > 1e-9)
        throw std::runtime_error("Invalid physics rollout: simulation time did not advance by one timestep");
    for (int k = 0; k < sim.model->nq; k++) {
        if (!std::isfinite(sim.data->qpos[k]))
            throw std::runtime_error("Invalid physics rollout: nonfinite qpos");
    }
    for (int k = 0; k < sim.model->nv; k++) {
        if (!std::isfinite(sim.data->qvel[k]) || !std::isfinite(sim.data->qacc[k]))
            throw std::runtime_error("Invalid physics rollout: nonfinite qvel or qacc");
    }
}

void CheckedStep(Simulation& sim) {
    double previousTime = sim.data->time;
    try {
        sim.Step(false);
    } catch (const std::exception& error) {
        // The known height threshold is a physical failure only if the step
        // remained numerically valid. Policy/asset errors retain their cause.
        if (std::string(error.what()).rfind("G1 fell:", 0) == 0)
            CheckPhysicsNumerics(sim, previousTime);
        throw;
    }
    CheckPhysicsNumerics(sim, previousTime);
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 5 || argc > 6) return 2;
    try {
        const bool arms = std::string(argv[2]) == "arms";
        const std::string mode = argv[3];
        if (mode != "meta" && mode != "direct")
            throw std::runtime_error("Mode must be meta or direct");
        if (argc == 6 && std::string(argv[5]) != "full")
            throw std::runtime_error("Optional final argument must be full");
        const bool useMeta = mode == "meta";
        Simulation sim(argv[1]);
        MetaRetargeter meta(argv[1]);
        auto& g = meta.solver();
        auto* model = g.model();
        if (model->nq != 36 || model->nv != 35)
            throw std::runtime_error("Motion harness requires the G1 29-joint GMR model");
        std::unique_ptr<mjData, decltype(&mj_deleteData)> data(mj_makeData(model), mj_deleteData);
        if (!data) throw std::runtime_error("mj_makeData failed");

        TrackingFrame input;
        input.body.valid = input.body.supported = input.valid = true;
        input.location_flags = {15, 15, 15};
        input.body.confidence = 1;
        mj_resetData(model, data.get());
        data->qpos[2] = .8;
        for (auto side : {"left", "right"}) {
            int roll = Joint(model, std::string(side) + "_shoulder_roll_joint");
            int elbow = Joint(model, std::string(side) + "_elbow_joint");
            data->qpos[model->jnt_qposadr[roll]] = (std::string(side) == "left" ? 1 : -1)*kPi/2;
            data->qpos[model->jnt_qposadr[elbow]] = kPi/2;
        }
        mj_forward(model, data.get());
        FillSyntheticMeta(input, g, data.get(), true);

        double fps = 60;
        auto clip = arms ? ArmClip(sim, model, fps) : LoadClip(argv[2], fps);
        Qpos pose = clip.front();
        std::copy(pose.begin(), pose.end(), data->qpos);
        mj_forward(model, data.get());
        FillSyntheticMeta(input, g, data.get(), false);
        meta.Calibrate(sim.model, sim.data, input);

        Command command{}, initial{};
        initial[2] = sim.data->qpos[2];
        Qpos physical = PhysicalPose(sim, model);
        for (int k = 0; k < 29; k++) initial[k+6] = physical[k+7];
        std::ofstream out(argv[4]);
        if (!out) throw std::runtime_error("Cannot open output CSV");
        out << std::setprecision(17) << "wall_s,sim_s,height,tilt";
        for (int k = 0; k < 35; k++) out << ",cmd_" << k;
        AblationHeader(out);
        for (int k = 0; k < 36; k++) out << ",truth" << k;
        out << '\n';

        int frames = argc == 6 ? int(std::ceil((kStartupSeconds + clip.size()/fps)/kControlDt))
                               : (arms ? 2200 : 1400);
        Qpos last = clip.front();
        int previous = -1;
        double minHeight = 1, path = 0, xy[2] = {}, tilt = 0, legError = 0, rootError = 0;
        bool fell = false;
        int n = 0;
        for (; n < frames; n++) {
            double t = n*kControlDt;
            // Preserve the source cadence and a two-second initial hold.
            int sample = int(t*fps);
            int index = std::clamp(sample-int(kStartupSeconds*fps), 0, int(clip.size())-1);
            if (sample != previous) {
                pose = clip[index];
                std::copy(pose.begin(), pose.end(), data->qpos);
                mj_forward(model, data.get());
                FillSyntheticMeta(input, g, data.get(), false);
                input.body.time_ns = input.xr_time_ns = 1000000000LL + int64_t(sample/fps*1e9);
                if (useMeta) {
                    command = meta.Solve(input);
                    for (int k = 0; k < 12; k++)
                        legError = std::max(legError, std::abs(double(command[k+6])-pose[k+7]));
                    rootError = std::max(rootError, std::abs(double(command[2])-pose[2]));
                } else {
                    command = DirectCommand(model, last, pose, fps);
                }
                previous = sample;
                last = pose;
            }
            // Direct diagnostic reference is source truth, including policy
            // ticks that hold the same source sample.
            if (!useMeta) {
                std::copy(pose.begin(), pose.end(), g.data()->qpos);
                mj_forward(g.model(), g.data());
            }
            out << t << ',' << sim.data->time << ',' << sim.data->qpos[2] << ',' << TiltDegrees(sim.data);
            for (float value : command) out << ',' << value;
            AblationRow(out, sim, g, 1, t >= kStartupSeconds);
            for (double value : pose) out << ',' << value;
            out << '\n';

            Command blended = command;
            double u = std::min(1., t/kStartupSeconds);
            for (int k = 0; k < 35; k++) blended[k] = initial[k] + u*(command[k]-initial[k]);
            sim.SetWholeBodyReference(blended);
            try {
                for (int k = 0; k < 10; k++) CheckedStep(sim);
            } catch (const std::exception& error) {
                if (std::string(error.what()).rfind("G1 fell:", 0) != 0) throw;
                fell = true;
                break;
            }
            minHeight = std::min(minHeight, sim.data->qpos[2]);
            path += std::hypot(sim.data->qpos[0]-xy[0], sim.data->qpos[1]-xy[1]);
            xy[0] = sim.data->qpos[0];
            xy[1] = sim.data->qpos[1];
            tilt = std::max(tilt, TiltDegrees(sim.data));
        }
        if (!out) throw std::runtime_error("Cannot write output CSV");
        std::printf("%s seconds=%.2f fell=%d minz=%.4f path=%.3f finalXY=%.3f %.3f tilt=%.1f leg_ref_max=%.4f rootz_ref_max=%.4f\n",
                    argv[3], n*kControlDt, fell, minHeight, path, xy[0], xy[1], tilt, legError, rootError);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}

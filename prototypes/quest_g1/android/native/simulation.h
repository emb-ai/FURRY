#pragma once
#include <mujoco/mujoco.h>
#include <onnxruntime_cxx_api.h>
#include <array>
#include <memory>
#include <string>
#include <vector>

// CPU controller port from TWIST2 (MIT, Copyright 2025 Yanjie Ze).
// Shared by the Android app and the native desktop integration check.
class Simulation {
public:
    explicit Simulation(const std::string& assets, int physicsWorkers=2);
    ~Simulation();
    void Reset();
    void Step(bool demo = true, double grip = 0, double right_grip = -1);
    mjModel* model = nullptr;
    mjData* data = nullptr;
    double inference_ms = 0;
    int steps = 0;
    int physics_workers = 0;
    void SetArmReference(const std::array<float,29>& joints);
    void SetWholeBodyReference(const std::array<float,35>& reference);
    void PauseWholeBodyReference();
    const std::array<float,35>& WholeBodyReference()const{return whole_reference;}
    void ClearArmReference(){has_reference=false;}
private:
    std::unique_ptr<mjThreadPool, decltype(&mju_threadPoolDestroy)> physicsPool{nullptr, mju_threadPoolDestroy};
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "G1Quest"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    std::string input_name, output_name;
    std::array<int,29> qadr{}, vadr{}, aids{};
    std::array<float,29> last_action{}, target{};
    std::array<float,1270> history{};
    std::vector<int> hands;
    std::array<double,2> hand_grip{};
    bool has_whole_reference=false;
    std::array<float,35> whole_reference{};
    bool has_reference=false;
    std::array<float,14> arm_reference{};
};

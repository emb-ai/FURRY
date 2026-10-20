#pragma once
#include <mujoco/mujoco.h>
#include "tennis_task.h"
#include "standing_arms.h"
#include "collection_task.h"
#include "rubik_task.h"
#include <onnxruntime_cxx_api.h>
#include <array>
#include <memory>
#include <string>
#include <vector>
#include "policy_backend.h"
#include "groot_decoupled.h"
#include "body_motor_model.h"
#include "sonic.h"
#include "dexweave.h"

// CPU controller port from TWIST2 (MIT, Copyright 2025 Yanjie Ze).
// Shared by the Android app and the native desktop integration check.
class Simulation {
public:
    explicit Simulation(const std::string& assets, int physicsWorkers=2, const std::string& sceneName="lab", const std::string& policyPath="", int policyHzOverride=0);
    Simulation(const std::string& assets, int physicsWorkers, const std::string& sceneName, const questpolicy::RuntimePolicy& policy, int policyHzOverride=0);
    ~Simulation();
    void Reset();
    void NewSceneTrial(uint32_t seed){scene_seed=seed;Reset();}
    std::string CollectionStatus()const{return collection.Status(data);}
    std::string TennisStatus()const{return tennis.Status(data);}
    std::string RubikStatus()const{return rubik.Status(data);}
    bool RubikSolved()const{return rubik.Solved(data);}
    // This task input is archived in scene_state.csv userdata[7]. Saved-command
    // replay supplies it directly; live/retarget replay derives it from goals.
    void SetRubikRotationIntent(bool value){if(rubik.core>=0)data->userdata[RubikTask::RotationIntent]=value?1:0;}
    // Load and probe before committing; a failure preserves the active session.
    void SelectPolicy(const std::string& path);
    void SelectPolicy(const questpolicy::RuntimePolicy& policy);
    const std::string& Backend() const { return backend; }
    bool Standing() const { return standing; }
    void ResetArmCompensation(){standing_arms.Reset();}
    std::array<float,35> CorrectStandingArms(const std::array<float,35>& command,const std::vector<TrackedPose>& goals){
        if(rubik.core>=0&&goals.size()==14){
            rubik.ObserveGoal(goals[12].quaternion.data(),goals[13].quaternion.data(),data->time);
            SetRubikRotationIntent(rubik.HasRotationIntent(data->time));
        }
        if(!standing||goals.size()!=14)return command;
        return standing_arms.Correct(model,data,reference_data,command,{goals[12],goals[13]});
    }
    const grootdecoupled::Command& DecoupledCommand() const { return decoupled_command; }
    bool PrepareStep();
    bool Step(bool demo = true, double grip = 0, double right_grip = -1);
    mjModel* model = nullptr;
    mjData* data = nullptr;
    double inference_ms = 0;
    int steps = 0;
    double PlannerMs() const {return sonic_controller?sonic_controller->plannerMs:0;}
    double PlanAge() const {return sonic_controller?sonic_controller->planAge:0;}
    int PolicyHz() const { return policy_hz; }
    int physics_workers = 0;
    std::string scene_name;
    void SetArmReference(const std::array<float,29>& joints);
    void SetWholeBodyReference(const std::array<float,35>& reference);
    void PauseWholeBodyReference();
    void SetDexWeaveReference(const double* qpos);
    const std::array<double,36>& DexReference()const{return dex_reference;}
    const std::array<double,29>& DexVelocity()const{return dex_velocity;}
    const std::array<float,35>& WholeBodyReference()const{return whole_reference;}
    void ClearArmReference(){has_reference=false;}
public: // Evaluation-only access to initialize a recorded reset.
    std::string assets_directory;
    bool standing=false;
    StandingArms standing_arms;
    std::array<double,14> arm_gravity{};
    bodymotors::Profile original_body_motors{},decoupled_body_motors{};
    void StepDecoupled();
    void StepDexWeave();
    std::array<double,36> dex_reference{},dex_previous{};
    std::array<double,29> dex_velocity{};
    bool dex_has_reference=false,dex_has_previous=false;
    std::array<float,35> SonicCommand()const;
    std::unique_ptr<sonic::Controller> sonic_controller;
    grootdecoupled::Command MakeDecoupledCommand();
    std::string backend=questpolicy::Twist2;
    mjData* reference_data=nullptr;
    int torso_body=-1;
    std::array<float,grootdecoupled::Observation> decoupled_history{};
    std::array<float,grootdecoupled::Actions> decoupled_last_action{};
    grootdecoupled::Command decoupled_command;
    uint32_t scene_seed=0;
    CollectionTask collection;
    TennisTask tennis;
    RubikTask rubik;
    int policy_hz = 100;
    // 0 follows archived ONNX metadata; live Quest explicitly selects 50 Hz.
    const int policy_hz_override;
    std::unique_ptr<mjThreadPool, decltype(&mju_threadPoolDestroy)> physicsPool{nullptr, mju_threadPoolDestroy};
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "G1Quest"};
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session, secondary_session;
    std::string secondary_input_name, secondary_output_name;
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

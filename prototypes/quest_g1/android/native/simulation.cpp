#include "simulation.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace {
constexpr std::array<double,29> home = {-.2,0,0,.4,-.2,0, -.2,0,0,.4,-.2,0, 0,0,0,
                                      0,.4,0,1.2,0,0,0, 0,-.4,0,1.2,0,0,0};
constexpr std::array<double,29> kp = {100,100,100,150,40,40,100,100,100,150,40,40,150,150,150,
                                     40,40,40,40,4,4,4,40,40,40,40,4,4,4};
constexpr std::array<double,29> kd = {2,2,2,4,2,2,2,2,2,4,2,2,4,4,4,5,5,5,5,.2,.2,.2,5,5,5,5,.2,.2,.2};
std::vector<std::string> JointNames() {
    std::vector<std::string> out;
    for (auto side : {"left", "right"}) for (auto part : {"hip_pitch","hip_roll","hip_yaw","knee","ankle_pitch","ankle_roll"})
        out.push_back(std::string(side)+"_"+part+"_joint");
    for (auto part : {"yaw","roll","pitch"}) out.push_back(std::string("waist_")+part+"_joint");
    for (auto side : {"left", "right"}) for (auto part : {"shoulder_pitch","shoulder_roll","shoulder_yaw","elbow","wrist_roll","wrist_pitch","wrist_yaw"})
        out.push_back(std::string(side)+"_"+part+"_joint");
    return out;
}
}

Simulation::Simulation(const std::string& assets, int physicsWorkers, const std::string& sceneName, const std::string& policyPath, int policyHzOverride)
    : Simulation(assets,physicsWorkers,sceneName,questpolicy::RuntimePolicy{questpolicy::Twist2,policyPath.empty()?assets+"/policy.onnx":policyPath,""},policyHzOverride) {}

Simulation::Simulation(const std::string& assets,int physicsWorkers,const std::string& sceneName,
        const questpolicy::RuntimePolicy& policy,int policyHzOverride)
    : scene_name(sceneName), assets_directory(assets), policy_hz_override(policyHzOverride) {
    if(policyHzOverride!=0 && policyHzOverride!=50 && policyHzOverride!=100)
        throw std::runtime_error("Unsupported explicit policy frequency");
    if(physicsWorkers<0 || physicsWorkers>4)throw std::runtime_error("Invalid MuJoCo worker count");
    if(sceneName!="lab" && sceneName!="stand" && sceneName!="cup" && sceneName!="push_t" && sceneName!="tennis" && sceneName!="rubik" && sceneName!="wb_dex" && sceneName!="bimanual" && sceneName!="basket")
        throw std::runtime_error("Unknown Quest scene: "+sceneName);
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    SelectPolicy(policy);
    char error[2048] = {};
    const std::string sceneFile=sceneName=="lab"?"scene.xml":"scene-"+sceneName+".xml";
    model = mj_loadXML((assets+"/"+sceneFile).c_str(), nullptr, error, sizeof(error));
    if (!model) throw std::runtime_error(std::string("MuJoCo model: ")+error);
    data = mj_makeData(model);
    if (!data) throw std::runtime_error("mj_makeData failed");
    if(physicsWorkers){
        physicsPool.reset(mju_threadPoolCreate(physicsWorkers));
        mju_bindThreadPool(data,physicsPool.get());
        physics_workers=physicsWorkers;
    }
    auto names = JointNames();
    for (int i=0;i<29;i++) {
        int j=mj_name2id(model,mjOBJ_JOINT,names[i].c_str());
        int a=mj_name2id(model,mjOBJ_ACTUATOR,names[i].c_str());
        if(j<0 || a<0) throw std::runtime_error("Missing joint: "+names[i]);
        qadr[i]=model->jnt_qposadr[j]; vadr[i]=model->jnt_dofadr[j]; aids[i]=a;
        original_body_motors[i]=bodymotors::Capture(model,names[i]);
    }
    for(int i=0;i<model->nu;i++) {
        auto name=mj_id2name(model,mjOBJ_ACTUATOR,i);
        if(name && std::strstr(name,"hand_")) hands.push_back(i);
    }
    reference_data=mj_makeData(model);
    torso_body=mj_name2id(model,mjOBJ_BODY,"torso_link");
    if(!reference_data||torso_body<0)throw std::runtime_error("Missing decoupled torso kinematics");
    if(backend==questpolicy::GrootDecoupled || backend==questpolicy::Sonic){
        bodymotors::Apply(model,decoupled_body_motors);
        mj_setConst(model,data);
    }
    standing_arms.Initialize(model);
    collection.Initialize(model);
    tennis.Initialize(model);
    rubik.Initialize(model);
    Reset();
}
Simulation::~Simulation(){ sonic_controller.reset(); mj_deleteData(reference_data); mj_deleteData(data); mj_deleteModel(model); }
void Simulation::SelectPolicy(const std::string& path) {
    SelectPolicy(questpolicy::RuntimePolicy{questpolicy::Twist2,path,""});
}
void Simulation::SelectPolicy(const questpolicy::RuntimePolicy& policy) {
    policy.Validate();
    if(policy.backend==questpolicy::Sonic){
        if(policy_hz_override!=0&&policy_hz_override!=50)throw std::runtime_error("SONIC requires 50 Hz");
        auto motors=bodymotors::Load(policy.motorProfilePath.empty()?assets_directory+"/"+questpolicy::SonicMotorFile:policy.motorProfilePath,questpolicy::SonicMotorId,questpolicy::SonicMotorSha);
        if(model)bodymotors::ValidateMapping(model,motors);
        auto candidate=std::make_unique<sonic::Controller>(env,policy);
        sonic_controller=std::move(candidate);session.reset();secondary_session.reset();
        backend=policy.backend;standing=false;policy_hz=50;decoupled_body_motors=std::move(motors);
        if(model&&data){bodymotors::Apply(model,decoupled_body_motors);mj_setConst(model,data);Reset();}return;
    }
    const bool dex=policy.backend==questpolicy::DexWeave;
    const bool decoupled=policy.backend==questpolicy::GrootDecoupled;
    if((decoupled||dex) && policy_hz_override!=0 && policy_hz_override!=50)
        throw std::runtime_error("GR00T decoupled requires 50 Hz");
    const int inputWidth=dex?157:(decoupled?516:1432), outputWidth=decoupled?15:29;
    Ort::AllocatorWithDefaultOptions alloc;
    auto load=[&](const std::string& path,std::string& inputName,std::string& outputName){
        auto candidate=std::make_unique<Ort::Session>(env,path.c_str(),options);
        if(candidate->GetInputCount()!=1||candidate->GetOutputCount()!=1)
            throw std::runtime_error("Policy requires one input and one output");
        auto validate=[&](bool input,int width){
            auto type=input?candidate->GetInputTypeInfo(0):candidate->GetOutputTypeInfo(0);
            if(type.GetONNXType()!=ONNX_TYPE_TENSOR)throw std::runtime_error("Policy requires tensors");
            auto tensor=type.GetTensorTypeAndShapeInfo();auto shape=tensor.GetShape();
            if(tensor.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||shape.size()!=2||
               (shape[0]!=1&&shape[0]!=-1)||shape[1]!=width)
                throw std::runtime_error("Incompatible policy backend shape/type");
        };
        validate(true,inputWidth);validate(false,outputWidth);
        inputName=candidate->GetInputNameAllocated(0,alloc).get();
        outputName=candidate->GetOutputNameAllocated(0,alloc).get();
        std::vector<float> zeros(inputWidth);std::array<int64_t,2> shape{1,inputWidth};
        auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        auto tensor=Ort::Value::CreateTensor<float>(memory,zeros.data(),zeros.size(),shape.data(),shape.size());
        const char* inputs[]={inputName.c_str()};const char* outputs[]={outputName.c_str()};
        auto result=candidate->Run(Ort::RunOptions{nullptr},inputs,&tensor,1,outputs,1);
        if(result[0].GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{1,outputWidth})
            throw std::runtime_error("Invalid policy output shape");
        for(int i=0;i<outputWidth;i++)if(!std::isfinite(result[0].GetTensorData<float>()[i]))
            throw std::runtime_error("Invalid policy output");
        return candidate;
    };
    std::string nextInput,nextOutput,nextSecondaryInput,nextSecondaryOutput;
    auto candidate=load(policy.path,nextInput,nextOutput);
    std::unique_ptr<Ort::Session> secondaryCandidate;
    if(decoupled)secondaryCandidate=load(policy.secondaryPath,nextSecondaryInput,nextSecondaryOutput);
    bodymotors::Profile nextMotors;
    if(decoupled){
        nextMotors=bodymotors::Load(policy.motorProfilePath.empty()?assets_directory+"/"+bodymotors::File:policy.motorProfilePath);
        if(model)bodymotors::ValidateMapping(model,nextMotors);
    }
    int nextHz=50;
    if(!decoupled&&!dex){
        // Historical metadata-free TWIST2 episodes used 100 Hz; live overrides 50.
        auto rate=candidate->GetModelMetadata().LookupCustomMetadataMapAllocated("g1_policy_hz",alloc);
        std::string rateText=rate?rate.get():"100";
        if(rateText!="50"&&rateText!="100")throw std::runtime_error("Unsupported policy frequency");
        nextHz=policy_hz_override?policy_hz_override:(rateText=="50"?50:100);
    }
    // Both sessions are validated before changing any active backend or state.
    sonic_controller.reset();
    session=std::move(candidate);secondary_session=std::move(secondaryCandidate);
    input_name=std::move(nextInput);output_name=std::move(nextOutput);
    secondary_input_name=std::move(nextSecondaryInput);secondary_output_name=std::move(nextSecondaryOutput);
    backend=policy.backend;standing=policy.standing;policy_hz=nextHz;
    if(decoupled)decoupled_body_motors=std::move(nextMotors);
    if(model&&data){
        bodymotors::Apply(model,decoupled?decoupled_body_motors:original_body_motors);
        // Recompute model constants affected by motor inertia before restarting.
        mj_setConst(model,data);Reset();
    }
}
void Simulation::Reset() {
    mj_resetData(model,data);
    data->qpos[2]=.793; data->qpos[3]=1;
    const bool decoupled=backend==questpolicy::GrootDecoupled;
    const auto& defaults=backend==questpolicy::DexWeave?dexweave::Home:(backend==questpolicy::Sonic?sonic::home:(decoupled?grootdecoupled::Home:home));
    for(int i=0;i<29;i++){data->qpos[qadr[i]]=defaults[i]; target[i]=defaults[i];}
    if(backend==questpolicy::GrootDecoupled || backend==questpolicy::Sonic){data->qpos[qadr[18]]=data->qpos[qadr[25]]=1.2;}
    data->qpos[qadr[16]]=.2; data->qpos[qadr[23]]=-.2;
    if(standing&&(scene_name=="cup"||scene_name=="bimanual"||scene_name=="rubik")){
        data->qpos[0]=scene_name=="rubik"?.05:.18;
    }
    if(standing){
        // Start with relaxed arms beside the body in every scene. Operator
        // references take over normally after calibration; this is only rest.
        for(int h=0;h<2;h++){
            for(int j=0;j<7;j++)data->qpos[qadr[15+7*h+j]]=0;
            data->qpos[qadr[16+7*h]]=h?-.2:.2;
            data->qpos[qadr[18+7*h]]=1.4;
        }
    }
    if(backend!=questpolicy::Twist2)for(int i=0;i<29;i++)target[i]=data->qpos[qadr[i]];
    if(sonic_controller)sonic_controller->Reset();
    dex_has_reference=dex_has_previous=false;dex_velocity.fill(0);
    history.fill(0); last_action.fill(0); steps=0; hand_grip.fill(0);standing_arms.Reset();arm_gravity.fill(0);
    decoupled_history.fill(0);decoupled_last_action.fill(0);decoupled_command={};
    for(int i=0;i<14;i++)decoupled_command.arms[i]=data->qpos[qadr[15+i]];
    has_reference=false;has_whole_reference=false;whole_reference.fill(0);
    mj_forward(model,data);
    collection.Reset(model,data,scene_seed);
    tennis.Reset(model,data);
    tennis.SetSeed(model,data,scene_seed);
    rubik.Reset(model,data);
}
void Simulation::SetArmReference(const std::array<float,29>& joints){
    if(!has_reference)for(int k=0;k<14;k++)arm_reference[k]=data->qpos[qadr[15+k]];
    for(int k=0;k<14;k++){
        if(!std::isfinite(joints[15+k]))throw std::runtime_error("Invalid arm reference");
        arm_reference[k]+=std::clamp(joints[15+k]-arm_reference[k],-.02f,.02f);
    }
    has_reference=true;
}
void Simulation::SetWholeBodyReference(const std::array<float,35>& reference){
    for(double v:reference)if(!std::isfinite(v))throw std::runtime_error("Nonfinite GMR reference");
    whole_reference=reference;has_whole_reference=true;

}
void Simulation::PauseWholeBodyReference(){
    SetRubikRotationIntent(false);rubik.intentHistory.clear();rubik.rotationIntent=false;
    standing_arms.Reset();
    dex_has_previous=false;dex_velocity.fill(0);
    if(has_whole_reference)whole_reference[0]=whole_reference[1]=whole_reference[5]=0;
}
std::array<float,35> Simulation::SonicCommand()const{
    auto command=whole_reference;
    if(!has_whole_reference){command.fill(0);command[2]=.793f;for(int k=0;k<29;k++)command[6+k]=sonic::home[k];command[24]=command[31]=1.2f;
        if(has_reference)std::copy(arm_reference.begin(),arm_reference.end(),command.begin()+21);}
    return command;
}
bool Simulation::PrepareStep(){
    return !sonic_controller||steps%20!=0||sonic_controller->Prepare(data,qadr,SonicCommand());
}
bool Simulation::Step(bool demo,double grip,double right_grip) {
    if(!PrepareStep())return false;
    if(steps%20==0 && backend==questpolicy::DexWeave)StepDexWeave();
    if(steps%20==0 && sonic_controller){
        auto command=SonicCommand();
        auto start=std::chrono::steady_clock::now();
        target=sonic_controller->Update(model,data,reference_data,qadr,vadr,command);
        inference_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    }
    if(steps%(1000/policy_hz)==0 && backend==questpolicy::GrootDecoupled)StepDecoupled();
    if(steps%(1000/policy_hz)==0 && backend==questpolicy::Twist2) {
        std::array<float,1432> obs{};
        obs[2]=.8f;
        for(int i=0;i<29;i++) obs[6+i]=home[i];
        if(has_reference){
            for(int k=0;k<14;k++)obs[21+k]=arm_reference[k];
        }else if(demo) {
            double wave=.5*(1-std::cos(2*3.141592653589793*data->time/6));
            obs[21]-=.35*wave; obs[28]-=.35*wave;
            obs[24]+=.3*wave; obs[31]+=.3*wave;
        }
        if(has_whole_reference)std::copy(whole_reference.begin(),whole_reference.end(),obs.begin());
        for(int i=0;i<3;i++) obs[35+i]=data->qvel[3+i]*.25;
        double w=data->qpos[3],x=data->qpos[4],y=data->qpos[5],z=data->qpos[6];
        obs[38]=std::atan2(2*(w*x+y*z),1-2*(x*x+y*y));
        obs[39]=std::asin(std::clamp(2*(w*y-z*x),-1.,1.));
        for(int i=0;i<29;i++) {
            obs[40+i]=data->qpos[qadr[i]]-home[i];
            obs[69+i]=(i==4 || i==5 || i==10 || i==11) ? 0 : data->qvel[vadr[i]]*.05;
            obs[98+i]=last_action[i];
        }
        std::copy(history.begin(),history.end(),obs.begin()+127);
        std::copy_n(obs.begin(),35,obs.begin()+1397);
        std::move(history.begin()+127,history.end(),history.begin());
        std::copy_n(obs.begin(),127,history.end()-127);
        std::array<int64_t,2> shape{1,1432};
        auto mem=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        auto input=Ort::Value::CreateTensor<float>(mem,obs.data(),obs.size(),shape.data(),2);
        const char* ins[]={input_name.c_str()}; const char* outs[]={output_name.c_str()};
        auto start=std::chrono::steady_clock::now();
        auto outputs=session->Run(Ort::RunOptions{nullptr},ins,&input,1,outs,1);
        inference_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        auto action=outputs[0].GetTensorData<float>();
        for(int i=0;i<29;i++) {
            if(!std::isfinite(action[i])) throw std::runtime_error("Policy output is not finite");
            last_action[i]=action[i]; target[i]=home[i]+.5*std::clamp(action[i],-10.f,10.f);
        }
    }
    for(int i=0;i<29;i++) {
        if(backend==questpolicy::DexWeave){
            data->ctrl[aids[i]]=std::clamp((target[i]-data->qpos[qadr[i]])*dexweave::Kp(i)-data->qvel[vadr[i]]*dexweave::Kd(i),-dexweave::Effort[i],dexweave::Effort[i]);
        }else if(backend==questpolicy::GrootDecoupled){
            // Author motor PD; MuJoCo applies the model's actuator force limits.
            data->ctrl[aids[i]]=(target[i]-data->qpos[qadr[i]])*grootdecoupled::Kp[i]-data->qvel[vadr[i]]*grootdecoupled::Kd[i]+(standing&&i>=15?arm_gravity[i-15]:0);
        }else if(sonic_controller)data->ctrl[aids[i]]=(target[i]-data->qpos[qadr[i]])*sonic::kp[i]-data->qvel[vadr[i]]*sonic::kd[i];
        else data->ctrl[aids[i]]=std::clamp((target[i]-data->qpos[qadr[i]])*kp[i]-data->qvel[vadr[i]]*kd[i],-kp[i],kp[i]);
    }
    if(backend==questpolicy::Twist2&&scene_name=="rubik"&&has_whole_reference)for(int i=15;i<29;i++){
        // Controller-driven manipulation uses direct joint PD for the arms;
        // TWIST2 still controls legs/waist. The object remains free/contact-only.
        bool wrist=(i>=19&&i<=21)||i>=26;
        double manipulationKp=wrist?25:kp[i],manipulationKd=wrist?.6:kd[i];
        double torque=(whole_reference[6+i]-data->qpos[qadr[i]])*manipulationKp-data->qvel[vadr[i]]*manipulationKd+data->qfrc_bias[vadr[i]];
        data->ctrl[aids[i]]=std::clamp(torque,-kp[i],kp[i]);
    }
    if(right_grip<0)right_grip=grip;
    hand_grip[0]+=std::clamp(grip-hand_grip[0],-.002,.002);
    hand_grip[1]+=std::clamp(right_grip-hand_grip[1],-.002,.002);
    tennis.UpdateGrip(model,data,grip,right_grip);
    for(int aid:hands) {
        std::string name=mj_id2name(model,mjOBJ_ACTUATOR,aid);
        double sign=name.rfind("left",0)==0 ? 1 : -1;
        double target=0;
        if(name.find("thumb_0")==std::string::npos) {
            if(name.find("thumb")!=std::string::npos) target=sign*(name.find("_1_")!=std::string::npos ? 1. : 1.74);
            else target=-sign*(name.find("_0_")!=std::string::npos ? 1.57 : 1.74);
        }
        int j=model->actuator_trnid[aid*2];
        double desired=target*hand_grip[sign>0?0:1];
        double control=tennis.HandControl(data,name,desired);
        data->ctrl[aid]=std::clamp(control,model->jnt_range[2*j],model->jnt_range[2*j+1]);
    }
    tennis.Begin(data);
    rubik.Begin(model,data,grip,right_grip);
    mj_step(model,data);
    tennis.Finish(model,data);
    rubik.Finish(model,data);
    collection.Finish(model,data);
    steps++;
    // Floor-placement models explicitly permit a deeper squat. Keep the
    // historical physics guard for legacy models and other scenes.
    const double minimumHeight=std::min(.35,collection.fallHeight);
    if(!std::isfinite(data->qpos[2]) || data->qpos[2]<minimumHeight) throw std::runtime_error("G1 fell: reset required");
    return true;
}

grootdecoupled::Command Simulation::MakeDecoupledCommand() {
    auto command=decoupled_command;
    if(has_reference)command.arms=arm_reference;
    if(!has_whole_reference)return command;
    command.velocity={whole_reference[0],whole_reference[1],whole_reference[5]};
    command.height=whole_reference[2];
    std::copy_n(whole_reference.begin()+21,14,command.arms.begin());
    // Derive torso orientation through the actual model, in yaw-only pelvis axes.
    // Reference legs are deliberately absent: they cannot prescribe RL foot motion.
    mj_resetData(model,reference_data);
    const mjtNum rollAxis[3]={1,0,0},pitchAxis[3]={0,1,0};mjtNum roll[4],pitch[4];
    mju_axisAngle2Quat(roll,rollAxis,whole_reference[3]);
    mju_axisAngle2Quat(pitch,pitchAxis,whole_reference[4]);
    mju_mulQuat(reference_data->qpos+3,pitch,roll);
    for(int i=0;i<3;i++)reference_data->qpos[qadr[12+i]]=whole_reference[18+i];
    mj_kinematics(model,reference_data);
    const auto* rotation=reference_data->xmat+9*torso_body;
    command.torsoRpy={float(std::atan2(rotation[7],rotation[8])),
                      float(std::asin(std::clamp(-rotation[6],-1.,1.))),
                      float(std::atan2(rotation[3],rotation[0]))};
    return command;
}
void Simulation::StepDecoupled() {
    decoupled_command=MakeDecoupledCommand();
    // Explicit manipulation mode: the Balance actor receives zero navigation.
    // Height, torso and both arms remain controlled by the operator.
    if(standing)decoupled_command.velocity.fill(0);
    std::array<double,29> q{},dq{};std::array<double,3> omega{};std::array<double,4> quat{};
    for(int i=0;i<29;i++){q[i]=data->qpos[qadr[i]];dq[i]=data->qvel[vadr[i]];}
    std::copy_n(data->qvel+3,3,omega.begin());std::copy_n(data->qpos+3,4,quat.begin());
    grootdecoupled::Append(decoupled_history,grootdecoupled::Observe(decoupled_command,q,dq,omega,quat,decoupled_last_action));
    const bool walking=grootdecoupled::Walking(decoupled_command);
    auto& active=walking?secondary_session:session;
    const char* ins[]={(walking?secondary_input_name:input_name).c_str()};
    const char* outs[]={(walking?secondary_output_name:output_name).c_str()};
    std::array<int64_t,2> shape{1,grootdecoupled::Observation};
    auto mem=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    auto input=Ort::Value::CreateTensor<float>(mem,decoupled_history.data(),decoupled_history.size(),shape.data(),2);
    auto start=std::chrono::steady_clock::now();
    auto outputs=active->Run(Ort::RunOptions{nullptr},ins,&input,1,outs,1);
    inference_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    const auto* action=outputs[0].GetTensorData<float>();
    for(int i=0;i<15;i++){
        if(!std::isfinite(action[i]))throw std::runtime_error("Decoupled policy output is not finite");
        decoupled_last_action[i]=action[i];target[i]=grootdecoupled::Home[i]+.25f*action[i];
    }
    std::copy(decoupled_command.arms.begin(),decoupled_command.arms.end(),target.begin()+15);
    if(standing){
        // Recompute gravity at the 50 Hz control boundary from current qpos.
        // Never consume mj_step's stale pre-integration bias cache: rendering
        // and calibration may refresh that cache at different times in replay.
        std::copy_n(data->qpos,model->nq,reference_data->qpos);
        mju_zero(reference_data->qvel,model->nv);
        mj_kinematics(model,reference_data);mj_comPos(model,reference_data);mj_comVel(model,reference_data);
        mj_rne(model,reference_data,0,reference_data->qfrc_bias);
        for(int i=0;i<14;i++)arm_gravity[i]=reference_data->qfrc_bias[vadr[15+i]];
    }
}

void Simulation::SetDexWeaveReference(const double* qpos){
    for(int i=0;i<36;i++)if(!std::isfinite(qpos[i]))throw std::runtime_error("Invalid DexWeave reference");
    const double norm=mju_norm(qpos+3,4);
    if(std::abs(norm-1)>1e-4)throw std::runtime_error("Invalid DexWeave reference quaternion");
    std::copy_n(qpos,36,dex_reference.begin());dex_has_reference=true;
}
void Simulation::StepDexWeave(){
    if(!dex_has_reference){
        std::copy_n(data->qpos,7,dex_reference.begin());
        for(int i=0;i<29;i++)dex_reference[7+i]=data->qpos[qadr[i]];
        dex_has_reference=true;
    }
    for(int i=0;i<29;i++)dex_velocity[i]=dex_has_previous?(dex_reference[7+i]-dex_previous[7+i])/.02:0.;
    dex_previous=dex_reference;dex_has_previous=true;
    std::copy_n(data->qpos,model->nq,reference_data->qpos);
    std::copy_n(dex_reference.data(),7,reference_data->qpos);
    for(int i=0;i<29;i++)reference_data->qpos[qadr[i]]=dex_reference[7+i];
    mj_kinematics(model,reference_data);
    // mj_step's cached body transforms precede integration; refresh at the control boundary.
    mj_kinematics(model,data);
    auto obs=dexweave::Observe(data,reference_data,torso_body,qadr,vadr,dex_velocity,last_action);
    std::array<int64_t,2> shape{1,157};auto mem=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    auto input=Ort::Value::CreateTensor<float>(mem,obs.data(),obs.size(),shape.data(),2);
    const char* ins[]={input_name.c_str()};const char* outs[]={output_name.c_str()};
    auto start=std::chrono::steady_clock::now();
    auto output=session->Run(Ort::RunOptions{nullptr},ins,&input,1,outs,1);
    inference_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    dexweave::Action(output[0].GetTensorData<float>(),target,last_action);
}

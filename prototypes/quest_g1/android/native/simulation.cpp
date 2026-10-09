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

Simulation::Simulation(const std::string& assets, int physicsWorkers, const std::string& sceneName, const std::string& policyPath)
    : scene_name(sceneName) {
    if(physicsWorkers<0 || physicsWorkers>4)throw std::runtime_error("Invalid MuJoCo worker count");
    if(sceneName!="lab" && sceneName!="stand" && sceneName!="cup" && sceneName!="push_t")
        throw std::runtime_error("Unknown Quest scene: "+sceneName);
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    SelectPolicy(policyPath.empty()?assets+"/policy.onnx":policyPath);
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
    }
    for(int i=0;i<model->nu;i++) {
        auto name=mj_id2name(model,mjOBJ_ACTUATOR,i);
        if(name && std::strstr(name,"hand_")) hands.push_back(i);
    }
    Reset();
}
Simulation::~Simulation(){ mj_deleteData(data); mj_deleteModel(model); }
void Simulation::SelectPolicy(const std::string& path) {
    auto candidate=std::make_unique<Ort::Session>(env,path.c_str(),options);
    if(candidate->GetInputCount()!=1||candidate->GetOutputCount()!=1)
        throw std::runtime_error("Policy requires one input and one output");
    auto validate=[&](bool input,int width){
        auto type=input?candidate->GetInputTypeInfo(0):candidate->GetOutputTypeInfo(0);
        if(type.GetONNXType()!=ONNX_TYPE_TENSOR)throw std::runtime_error("Policy requires tensors");
        auto tensor=type.GetTensorTypeAndShapeInfo();auto shape=tensor.GetShape();
        if(tensor.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||shape.size()!=2||
           (shape[0]!=1&&shape[0]!=-1)||shape[1]!=width)
            throw std::runtime_error("Incompatible Quest policy shape/type");
    };
    validate(true,1432);validate(false,29);
    Ort::AllocatorWithDefaultOptions alloc;
    std::string nextInput=candidate->GetInputNameAllocated(0,alloc).get();
    std::string nextOutput=candidate->GetOutputNameAllocated(0,alloc).get();
    std::array<float,1432> zeros{};std::array<int64_t,2> shape{1,1432};
    auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    auto tensor=Ort::Value::CreateTensor<float>(memory,zeros.data(),zeros.size(),shape.data(),shape.size());
    const char* inputs[]={nextInput.c_str()};const char* outputs[]={nextOutput.c_str()};
    auto result=candidate->Run(Ort::RunOptions{nullptr},inputs,&tensor,1,outputs,1);
    if(result[0].GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{1,29})
        throw std::runtime_error("Invalid policy output shape");
    for(int i=0;i<29;i++)if(!std::isfinite(result[0].GetTensorData<float>()[i]))
        throw std::runtime_error("Invalid policy output");
    session=std::move(candidate);input_name=std::move(nextInput);output_name=std::move(nextOutput);
    if(model&&data)Reset();
}
void Simulation::Reset() {
    mj_resetData(model,data);
    data->qpos[2]=.793; data->qpos[3]=1;
    for(int i=0;i<29;i++){data->qpos[qadr[i]]=home[i]; target[i]=home[i];}
    data->qpos[qadr[16]]=.2; data->qpos[qadr[23]]=-.2;
    history.fill(0); last_action.fill(0); steps=0; hand_grip.fill(0);
    has_reference=false;has_whole_reference=false;whole_reference.fill(0);
    mj_forward(model,data);
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
    if(has_whole_reference)whole_reference[0]=whole_reference[1]=whole_reference[5]=0;
}
void Simulation::Step(bool demo,double grip,double right_grip) {
    if(steps%10==0) {
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
    for(int i=0;i<29;i++) data->ctrl[aids[i]]=std::clamp((target[i]-data->qpos[qadr[i]])*kp[i]-data->qvel[vadr[i]]*kd[i],-kp[i],kp[i]);
    if(right_grip<0)right_grip=grip;
    hand_grip[0]+=std::clamp(grip-hand_grip[0],-.002,.002);
    hand_grip[1]+=std::clamp(right_grip-hand_grip[1],-.002,.002);
    for(int aid:hands) {
        std::string name=mj_id2name(model,mjOBJ_ACTUATOR,aid);
        double sign=name.rfind("left",0)==0 ? 1 : -1;
        double target=0;
        if(name.find("thumb_0")==std::string::npos) {
            if(name.find("thumb")!=std::string::npos) target=sign*(name.find("_1_")!=std::string::npos ? 1. : 1.74);
            else target=-sign*(name.find("_0_")!=std::string::npos ? 1.57 : 1.74);
        }
        int j=model->actuator_trnid[aid*2];
        data->ctrl[aid]=std::clamp(target*hand_grip[sign>0?0:1],model->jnt_range[2*j],model->jnt_range[2*j+1]);
    }
    mj_step(model,data); steps++;
    if(!std::isfinite(data->qpos[2]) || data->qpos[2]<.35) throw std::runtime_error("G1 fell: reset required");
}

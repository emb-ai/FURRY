#pragma once
// SONIC VR3 contract port. Source: NVlabs/GR00T-WholeBodyControl b042411f.
// The planner owns no MuJoCo pointers and runs separately from physics on Quest.
#include "sonic_constants.h"
#include "policy_backend.h"
#include <mujoco/mujoco.h>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <memory>
#include <vector>

namespace sonic {
struct PlannerPending : std::exception { const char* what()const noexcept override{return "SONIC waiting for planner";} };
using Pose=std::array<double,36>;
struct Plan {std::vector<Pose> poses; double start=0,ms=0;};
struct Request {std::array<float,144> context{};std::array<float,3> move{},face{};float speed=.01f;int64_t mode=0;double time=0;};
inline void Inverse(double* dst,const double* q){mju_negQuat(dst,q);}
inline std::vector<float> Run(Ort::Session& s,const std::vector<float>& values,int width) {
    Ort::AllocatorWithDefaultOptions a;auto in=s.GetInputNameAllocated(0,a),out=s.GetOutputNameAllocated(0,a);
    int64_t shape[2]={1,int64_t(values.size())};auto mem=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    auto t=Ort::Value::CreateTensor<float>(mem,const_cast<float*>(values.data()),values.size(),shape,2);
    const char* ins[]={in.get()};const char* outs[]={out.get()};auto y=s.Run(Ort::RunOptions{nullptr},ins,&t,1,outs,1);
    if(y[0].GetTensorTypeAndShapeInfo().GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||y[0].GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{1,width})throw std::runtime_error("SONIC output shape");
    const auto* p=y[0].GetTensorData<float>();std::vector<float> result(p,p+width);
    for(float v:result)if(!std::isfinite(v))throw std::runtime_error("SONIC nonfinite output");return result;
}
inline void Validate(Ort::Session& s,int in,int out){
    if(s.GetInputCount()!=1||s.GetOutputCount()!=1)throw std::runtime_error("SONIC tensor count");
    for(bool input:{true,false}){auto t=input?s.GetInputTypeInfo(0):s.GetOutputTypeInfo(0);auto tensor=t.GetTensorTypeAndShapeInfo();
        if(tensor.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||tensor.GetShape()!=std::vector<int64_t>{1,input?in:out})throw std::runtime_error("SONIC contract mismatch");}
    Run(s,std::vector<float>(in),out);
}
class Controller {
    std::unique_ptr<Ort::Session> encoder,decoder,planner;
    std::future<Plan> pending;
    bool asynchronous;
    Plan plan;
    double requested=-1,pinnedTime=0;bool planPinned=false;
    std::array<std::array<double,93>,10> history{}; // omega, q, dq, previous action, gravity
    std::array<double,29> previous{};
    Plan Predict(Request r){
        auto start=std::chrono::steady_clock::now();
        auto mem=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);std::vector<Ort::Value> ts;
        auto f=[&](float* data,size_t n,std::initializer_list<int64_t> shape){std::vector<int64_t> s(shape);ts.push_back(Ort::Value::CreateTensor<float>(mem,data,n,s.data(),s.size()));};
        auto i=[&](int64_t* data,size_t n,std::initializer_list<int64_t> shape){std::vector<int64_t> s(shape);ts.push_back(Ort::Value::CreateTensor<int64_t>(mem,data,n,s.data(),s.size()));};
        int64_t seed=11,specific=0,allowed[11]={1,1,1,1,1,1,0,0,0,0,0};float positions[12]={},headings[4]={},height=-1;
        f(r.context.data(),144,{1,4,36});f(&r.speed,1,{1});i(&r.mode,1,{1});f(r.move.data(),3,{1,3});f(r.face.data(),3,{1,3});
        i(&seed,1,{1});i(&specific,1,{1,1});f(positions,12,{1,4,3});f(headings,4,{1,4});i(allowed,11,{1,11});f(&height,1,{1});
        const char* ins[]={"context_mujoco_qpos","target_vel","mode","movement_direction","facing_direction","random_seed","has_specific_target","specific_target_positions","specific_target_headings","allowed_pred_num_tokens","height"};
        const char* outs[]={"mujoco_qpos","num_pred_frames"};auto y=planner->Run(Ort::RunOptions{nullptr},ins,ts.data(),ts.size(),outs,2);
        if(y[0].GetTensorTypeAndShapeInfo().GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||y[0].GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{1,64,36}||y[1].GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{1}||y[1].GetTensorTypeAndShapeInfo().GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)throw std::runtime_error("SONIC planner output shape");
        int n=y[1].GetTensorData<int32_t>()[0];if(n<4||n>64)throw std::runtime_error("SONIC planner frame count");
        Plan result;result.start=r.time;result.poses.resize(n);const float* p=y[0].GetTensorData<float>();
        for(int j=0;j<n;j++)for(int k=0;k<36;k++){double v=p[j*36+k];if(!std::isfinite(v))throw std::runtime_error("SONIC planner nonfinite");result.poses[j][k]=v;}
        for(auto& q:result.poses){double norm=mju_norm(q.data()+3,4);if(norm<.5||norm>1.5)throw std::runtime_error("SONIC planner invalid quaternion");mju_normalize4(q.data()+3);}
        result.ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();return result;
    }
public:
    double plannerMs=0,planAge=0;
    Controller(Ort::Env& env,const questpolicy::RuntimePolicy& policy):asynchronous(policy.asyncPlanner){
        Ort::SessionOptions opt;opt.SetIntraOpNumThreads(1);opt.SetInterOpNumThreads(1);
        encoder=std::make_unique<Ort::Session>(env,policy.path.c_str(),opt);Validate(*encoder,1762,64);
        decoder=std::make_unique<Ort::Session>(env,policy.secondaryPath.c_str(),opt);Validate(*decoder,994,29);
        planner=std::make_unique<Ort::Session>(env,policy.plannerPath.c_str(),opt);
        if(planner->GetInputCount()!=11||planner->GetOutputCount()!=2)throw std::runtime_error("SONIC planner tensor count");
        Request r;for(int j=0;j<4;j++){r.context[j*36+2]=.793f;r.context[j*36+3]=1;for(int k=0;k<29;k++)r.context[j*36+7+k]=home[k];}r.face[0]=1;
        Predict(r);Reset(); // Probe all three actual models before committing a selection.
    }
    ~Controller(){if(pending.valid())pending.wait();}
    void Reset(){if(pending.valid()){pending.wait();try{pending.get();}catch(...){}}plan={};requested=-1;planPinned=false;previous.fill(0);history={};for(auto& h:history)h[92]=-1;plannerMs=planAge=0;}
    Pose At(double time)const{
        if(plan.poses.empty())throw std::runtime_error("SONIC missing plan");
        double x=std::clamp((time-plan.start)*30.,0.,double(plan.poses.size()-1));size_t a=size_t(x),b=std::min(a+1,plan.poses.size()-1);double w=x-a;Pose q;
        for(int k=0;k<36;k++)q[k]=plan.poses[a][k]*(1-w)+plan.poses[b][k]*w;
        const auto* qa=plan.poses[a].data()+3;const auto* qb=plan.poses[b].data()+3;
        double dot=mju_dot(qa,qb,4),sign=dot<0?-1:1;dot=std::clamp(std::abs(dot),0.,1.);double wa=1-w,wb=w;
        if(dot<.9995){double theta=std::acos(dot);wa=std::sin((1-w)*theta)/std::sin(theta);wb=std::sin(w*theta)/std::sin(theta);}
        for(int k=0;k<4;k++)q[3+k]=wa*qa[k]+wb*sign*qb[k];mju_normalize4(q.data()+3);return q;
    }
    bool PlanFor(const Pose& state,const std::array<float,35>& cmd,double now){
        // PrepareStep and Update belong to the same physical policy tick. A
        // future completing between them must not replace the already approved
        // plan after the runtime has recorded that tick's source frame.
        if(planPinned&&now==pinnedTime)return true;
        if(pending.valid()&&pending.wait_for(std::chrono::seconds(0))==std::future_status::ready){plan=pending.get();plannerMs=plan.ms;}
        const double remaining=plan.poses.empty()?0:plan.start+(plan.poses.size()-1)/30.-now;
        if(!pending.valid()&&(plan.poses.empty()||now-requested>=.2-1e-8||(asynchronous&&remaining<=.2))){
            Request r;r.time=now;Pose last;
            for(int j=0;j<4;j++){last=plan.poses.empty()?state:At(now+j/30.);for(int k=0;k<36;k++)r.context[j*36+k]=last[k];}
            double local[3]={cmd[0],cmd[1],0},world[3];mju_rotVecQuat(world,local,state.data()+3);double speed=std::hypot(world[0],world[1]);
            r.speed=std::max(speed,.01);r.mode=speed<.05?0:(speed<.5?1:2);
            r.move={float(world[0]/std::max(speed,1e-8)),float(world[1]/std::max(speed,1e-8)),0};
            auto* q=last.data()+3;double yaw=std::atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]*q[2]+q[3]*q[3]))+cmd[5]*.3;
            r.face={float(std::cos(yaw)),float(std::sin(yaw)),0};requested=now;
            if(asynchronous)pending=std::async(std::launch::async,[this,r]{return Predict(r);});
            else {plan=Predict(r);plannerMs=plan.ms;}
        }
        planAge=plan.poses.empty()?0:now-plan.start;
        // Backpressure simulation time instead of advancing on an exhausted
        // plan. Rendering/input stay live while the asynchronous result arrives.
        if(plan.poses.empty()||planAge>(plan.poses.size()-1)/30.-.02){
            if(asynchronous&&pending.valid())return false;
            throw std::runtime_error("SONIC planner late: trajectory exhausted");
        }
        pinnedTime=now;planPinned=true;return true;
    }
    bool Prepare(const mjData* data,const std::array<int,29>& qa,const std::array<float,35>& cmd){
        Pose state;std::copy_n(data->qpos,7,state.begin());for(int k=0;k<29;k++)state[7+k]=data->qpos[qa[k]];
        return PlanFor(state,cmd,data->time);
    }
    std::vector<float> EncoderObservation(const mjModel* model,mjData* ref,const Pose& state,const std::array<int,29>& qa,const std::array<float,35>& cmd,double now){
        std::vector<float> obs(1762);obs[0]=1; // teleop_mode scalar id, not one-hot
        double inv[4],delta[4],mat[9];Inverse(inv,state.data()+3);auto future=At(now);mju_mulQuat(delta,inv,future.data()+3);mju_quat2Mat(mat,delta);
        for(int row=0;row<3;row++)for(int col=0;col<2;col++)obs[595+row*2+col]=mat[row*3+col];
        for(int j=0;j<10;j++){auto a=At(now+j*.1),b=At(now+j*.1+.02);for(int k=0;k<12;k++){obs[661+j*12+k]=a[7+k];obs[781+j*12+k]=(b[7+k]-a[7+k])/.02;}}
        mj_resetData(model,ref);ref->qpos[2]=cmd[2];double roll[4],pitch[4];const double ax[3]={1,0,0},ay[3]={0,1,0};
        mju_axisAngle2Quat(roll,ax,cmd[3]);mju_axisAngle2Quat(pitch,ay,cmd[4]);mju_mulQuat(ref->qpos+3,pitch,roll);
        // Only waist/arms affect these FK targets. Incoming virtual leg joints are ignored.
        for(int k=12;k<29;k++)ref->qpos[qa[k]]=cmd[6+k];mj_kinematics(model,ref);Inverse(inv,ref->qpos+3);
        const char* names[]={"left_wrist_yaw_link","right_wrist_yaw_link","torso_link"};const double offsets[3][3]={{.18,-.025,0},{.18,.025,0},{0,0,.35}};
        for(int k=0;k<3;k++){int bid=mj_name2id(model,mjOBJ_BODY,names[k]);if(bid<0)throw std::runtime_error("SONIC VR body missing");
            double p[3],local[3],q[4];mju_rotVecQuat(p,offsets[k],ref->xquat+4*bid);for(int j=0;j<3;j++)p[j]+=ref->xpos[3*bid+j]-ref->qpos[j];mju_rotVecQuat(local,p,inv);mju_mulQuat(q,inv,ref->xquat+4*bid);
            for(int j=0;j<3;j++)obs[901+k*3+j]=local[j];for(int j=0;j<4;j++)obs[910+k*4+j]=q[j];}
        return obs;
    }
    std::array<float,29> Update(const mjModel* model,mjData* data,mjData* ref,const std::array<int,29>& qa,const std::array<int,29>& va,const std::array<float,35>& cmd){
        Pose state;std::copy_n(data->qpos,7,state.begin());for(int k=0;k<29;k++)state[7+k]=data->qpos[qa[k]];
        if(!PlanFor(state,cmd,data->time))throw PlannerPending{};auto token=Run(*encoder,EncoderObservation(model,ref,state,qa,cmd,data->time),64);
        std::move(history.begin()+1,history.end(),history.begin());auto& h=history.back();
        for(int k=0;k<3;k++)h[k]=data->qvel[3+k];for(int k=0;k<29;k++){h[3+k]=data->qpos[qa[to_isaac[k]]]-home[to_isaac[k]];h[32+k]=data->qvel[va[to_isaac[k]]];h[61+k]=previous[k];}
        double inv[4],g[3]={0,0,-1};Inverse(inv,state.data()+3);mju_rotVecQuat(h.data()+90,g,inv);
        std::vector<float> obs=token;int offset=0;for(int width:{3,29,29,29,3}){for(const auto& frame:history)for(int k=0;k<width;k++)obs.push_back(frame[offset+k]);offset+=width;}
        auto action=Run(*decoder,obs,29);std::copy(action.begin(),action.end(),previous.begin());std::array<float,29> target;
        for(int k=0;k<29;k++)target[k]=home[k]+scale[k]*previous[to_mujoco[k]];return target;
    }
};
} // namespace sonic

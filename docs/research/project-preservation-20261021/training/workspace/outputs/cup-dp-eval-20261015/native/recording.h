#pragma once
#include "retarget.h"
#include "episode_manifest.h"
#include "body_motor_model.h"
#include "full_body_io.h"
#include "learning_capture.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>

// Caller serializes access. Files stay local to the application; no participant identifiers.
// A missing complete.json means interrupted/unfinalized, never a successful episode.
class EpisodeRecorder {
    std::ofstream input, frames, events, body, mimic, tennis, rubik, dexReference, sceneState;
    fullbody::Recorder fullBody;
    LearningCapture learning;
    std::filesystem::path directory;
    uint64_t inputs=0, samples=0;
    int64_t lastFlush=0;
    template<class T> static void Values(std::ostream& out,const T* p,int n){for(int i=0;i<n;i++)out<<','<<p[i];}
public:
    static int64_t Now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
    bool active=false;
    bool failed=false;
    std::string path()const{return directory.string();}
    void Start(const std::string& root,const mjModel* model,const std::string& metadata="recording_metadata.json",const std::string& policyFile="",bool catchUp=false,bool firstPerson=false,const std::string& secondaryPolicyFile="",bool visualCapture=true){
        if(active)return;
        failed=false;inputs=samples=0;
        auto stamp=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        directory=std::filesystem::path(root)/"recordings"/("episode-"+std::to_string(stamp));
        try{
            std::filesystem::create_directories(directory.parent_path());
            if(!std::filesystem::create_directory(directory))throw std::runtime_error("Episode already exists");
            std::ifstream source(std::filesystem::path(root)/metadata);
            if(!source)throw std::runtime_error("Missing episode metadata");
            std::string raw{std::istreambuf_iterator<char>(source),std::istreambuf_iterator<char>()};
            size_t open=raw.find_first_not_of(" \t\r\n"),next=raw.find_first_not_of(" \t\r\n",open+1);
            if(open==std::string::npos||raw[open]!='{'||next==std::string::npos)throw std::runtime_error("Invalid episode metadata");
            std::ofstream manifest;manifest.exceptions(std::ios::badbit|std::ios::failbit);manifest.open(directory/"manifest.json");
            manifest<<raw.substr(0,open+1)<<"\n\"catch_up\":"<<(catchUp?"true":"false")
                <<",\"full_body_schema\":1,\"first_person\":"<<(firstPerson?"true":"false")<<(raw[next]=='}'?"":",")<<raw.substr(open+1);
            manifest.close();
            const auto identity=episodereplay::LoadManifest(directory.string());
            if(identity.policyBackend==questpolicy::GrootDecoupled){
                const auto profilePath=std::filesystem::path(root)/bodymotors::File;
                bodymotors::ValidateApplied(model,bodymotors::Load(profilePath.string()));
                std::filesystem::copy_file(profilePath,directory/bodymotors::File);
            }
            if(identity.policyBackend==questpolicy::Sonic){
                const auto profilePath=std::filesystem::path(root)/questpolicy::SonicMotorFile;
                bodymotors::ValidateApplied(model,bodymotors::Load(profilePath.string(),questpolicy::SonicMotorId,questpolicy::SonicMotorSha));
                std::filesystem::copy_file(profilePath,directory/questpolicy::SonicMotorFile);
                std::filesystem::copy_file(std::filesystem::path(root)/identity.policyPlannerFile,directory/"policy-planner.onnx");
                questpolicy::VerifySha256((directory/"policy-planner.onnx").string(),identity.policyPlannerSha256);
            }
            if(identity.policyBackend==questpolicy::DexWeave){
                dexReference.open(directory/"dexweave-reference.csv");
                dexReference<<"step,sim_time";
                for(int i=0;i<36;i++)dexReference<<",q"<<i;
                for(int i=0;i<29;i++)dexReference<<",dq"<<i;
                dexReference<<'\n'<<std::setprecision(17);
            }
            if(!policyFile.empty())std::filesystem::copy_file(std::filesystem::path(root)/policyFile,directory/"policy.onnx");
            if(!secondaryPolicyFile.empty())std::filesystem::copy_file(std::filesystem::path(root)/secondaryPolicyFile,directory/"policy-secondary.onnx");
            for(auto* f:{&input,&frames,&events,&body,&mimic,&tennis,&rubik,&dexReference,&sceneState})f->exceptions(std::ios::badbit|std::ios::failbit);
            input.open(directory/"input.csv");frames.open(directory/"frames.csv");events.open(directory/"events.csv");
            input<<"sequence,xr_time_ns,receive_ns,valid,head_flags,left_flags,right_flags,left_active,right_active,grip_left,grip_right";
            for(auto name:{"head","left","right"})for(auto c:{"x","y","z","qw","qx","qy","qz"})input<<','<<name<<'_'<<c;
            input<<'\n'<<std::setprecision(17);
            frames<<"receive_ns,sequence,step,sim_time,reset,calibrate,apply_reference,valid,grip_left,grip_right,mode,ik_error,limited";
            for(int i=0;i<29;i++)frames<<",reference_"<<i;
            for(int i=0;i<model->nq;i++)frames<<",qpos_"<<i;
            for(int i=0;i<model->nv;i++)frames<<",qvel_"<<i;
            for(int i=0;i<model->nu;i++)frames<<",ctrl_"<<i;
            frames<<'\n'<<std::setprecision(17);
            events<<"receive_ns,sequence,event\n";
            body.open(directory/"body.csv");mimic.open(directory/"mimic.csv");
            body<<"sequence,time_ns,supported,valid,confidence,skeleton_version";
            for(int j=0;j<14;j++){body<<",flags_"<<j;for(auto kind:{"pose","rest"})for(int k=0;k<7;k++)body<<','<<kind<<'_'<<j<<'_'<<k;}
            body<<'\n'<<std::setprecision(17);
            mimic<<"sequence,step,sim_time";for(int k=0;k<35;k++)mimic<<",mimic_"<<k;mimic<<'\n'<<std::setprecision(17);
            fullBody.Start(directory);
            if((model->nuserdata==8||model->nuserdata==9||model->nuserdata==16) && mj_name2id(model,mjOBJ_BODY,"tennis_ball")>=0){
                tennis.open(directory/"tennis.csv");
                tennis<<"sequence,step,sim_time,armed,next_serve_s,active,racket_hit,own_bounces,serves,returns,misses";
                if(model->nuserdata>=9)tennis<<",holding_hand";
                if(model->nuserdata==16)tennis<<",seed,prepared_serve,launch_vx,launch_vy,launch_vz,bounce_x,bounce_y";
                tennis<<'\n'<<std::setprecision(17);
            }
            if(model->nuserdata==64&&mj_name2id(model,mjOBJ_BODY,"rubik_core")>=0){
                rubik.open(directory/"rubik.csv");rubik<<"sequence,step,sim_time";
                for(int i=0;i<64;i++)rubik<<",state_"<<i;
                rubik<<'\n'<<std::setprecision(17);
            }
            if(model->nuserdata||model->nmocap){
                sceneState.open(directory/"scene_state.csv");sceneState<<"sequence,step,sim_time";
                for(int i=0;i<model->nuserdata;i++)sceneState<<",userdata_"<<i;
                for(int i=0;i<model->nmocap*3;i++)sceneState<<",mocap_pos_"<<i;
                for(int i=0;i<model->nmocap*4;i++)sceneState<<",mocap_quat_"<<i;
                sceneState<<'\n'<<std::setprecision(17);
            }
            if(visualCapture)learning.Start(directory,model);
            active=true;Event("start",0);Flush();
        }catch(...){Abort();throw;}
    }
    void Event(const char* event,uint64_t sequence){if(active)learning.Event(event);if(active)events<<Now()<<','<<sequence<<','<<event<<'\n';}
    void Input(const TrackingFrame& f,float left,float right,int64_t received){
        if(!active)return;
        input<<f.sequence<<','<<f.xr_time_ns<<','<<received<<','<<f.valid;
        Values(input,f.location_flags.data(),3);
        input<<','<<f.hand_active[0]<<','<<f.hand_active[1]<<','<<left<<','<<right;
        for(const auto& pose:{f.head,f.hands[0],f.hands[1]}){Values(input,pose.position.data(),3);Values(input,pose.quaternion.data(),4);}
        input<<'\n';inputs++;
        body<<f.sequence<<','<<f.body.time_ns<<','<<f.body.supported<<','<<f.body.valid<<','<<f.body.confidence<<','<<f.body.skeleton_version;
        for(int j=0;j<14;j++){
            body<<','<<f.body.flags[j];
            for(const auto& p:{f.body.joints[j],f.body.rest[j]}){Values(body,p.position.data(),3);Values(body,p.quaternion.data(),4);}
        }
        body<<'\n';
        fullBody.Input(f,received);
    }
    void Frame(const mjModel* m,const mjData* d,int steps,const TrackingFrame& f,bool reset,bool calibrated,
               bool apply,bool valid,float left,float right,int mode,double error,bool limited,const std::array<float,29>& reference){
        if(!active)return;
        frames<<Now()<<','<<f.sequence<<','<<steps<<','<<d->time<<','<<reset<<','<<calibrated<<','<<apply<<','<<valid<<','<<left<<','<<right<<','<<mode<<','<<error<<','<<limited;
        Values(frames,reference.data(),29);Values(frames,d->qpos,m->nq);Values(frames,d->qvel,m->nv);Values(frames,d->ctrl,m->nu);
        frames<<'\n';samples++;
        learning.Frame(m,d,f.sequence,steps);
        if(sceneState.is_open()){
            sceneState<<f.sequence<<','<<steps<<','<<d->time;
            Values(sceneState,d->userdata,m->nuserdata);Values(sceneState,d->mocap_pos,m->nmocap*3);Values(sceneState,d->mocap_quat,m->nmocap*4);sceneState<<'\n';
        }
        if(tennis.is_open()){
            tennis<<f.sequence<<','<<steps<<','<<d->time;Values(tennis,d->userdata,m->nuserdata);tennis<<'\n';
        }
        if(rubik.is_open()){rubik<<f.sequence<<','<<steps<<','<<d->time;Values(rubik,d->userdata,m->nuserdata);rubik<<'\n';}
    }
    void Mimic(const TrackingFrame& f,int step,double time,const std::array<float,35>& value){
        if(!active)return;
        mimic<<f.sequence<<','<<step<<','<<time;Values(mimic,value.data(),35);mimic<<'\n';
    }
    void LearningGazeWorld(const double* direction,bool valid){learning.Gaze(direction,valid);}
    void LearningFinalState(const mjModel*m,const mjData*d){learning.FinalState(m,d);}
    void Flush(){if(active){learning.Flush();fullBody.Flush();input.flush();frames.flush();events.flush();body.flush();mimic.flush();if(tennis.is_open())tennis.flush();if(rubik.is_open())rubik.flush();if(dexReference.is_open())dexReference.flush();if(sceneState.is_open())sceneState.flush();lastFlush=Now();}}
    void Tick(){if(active && Now()-lastFlush>1000000000LL)Flush();}
    void DexReference(int step,double time,const std::array<double,36>& q,const std::array<double,29>& dq){
        if(!active||!dexReference.is_open())return;
        dexReference<<step<<','<<time;Values(dexReference,q.data(),36);Values(dexReference,dq.data(),29);dexReference<<'\n';
    }
    void Stop(const char* reason="user_stop"){
        if(!active)return;
        try{
            Event(reason,0);Flush();learning.Stop();fullBody.Close();input.close();frames.close();events.close();body.close();mimic.close();if(tennis.is_open())tennis.close();if(rubik.is_open())rubik.close();if(dexReference.is_open())dexReference.close();if(sceneState.is_open())sceneState.close();
            std::ofstream done;done.exceptions(std::ios::badbit|std::ios::failbit);
            done.open(directory/"complete.tmp");
            done<<"{\"schema_version\":1,\"reason\":\""<<reason<<"\",\"input_rows\":"<<inputs<<",\"frame_rows\":"<<samples<<"}\n";
            done.close();std::filesystem::rename(directory/"complete.tmp",directory/"complete.json");active=false;
        }catch(...){Abort();throw;}
    }
    void Abort()noexcept{
        active=false;failed=true;learning.Abort();fullBody.Abort();
        for(auto* f:{&input,&frames,&events,&body,&mimic,&tennis,&rubik,&dexReference,&sceneState}){f->exceptions(std::ios::goodbit);if(f->is_open())f->close();f->clear();}
    }
};

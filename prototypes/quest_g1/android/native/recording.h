#pragma once
#include "retarget.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>

// Caller serializes access. Files stay local to the application; no participant identifiers.
// A missing complete.json means interrupted/unfinalized, never a successful episode.
class EpisodeRecorder {
    std::ofstream input, frames, events, body, mimic;
    std::filesystem::path directory;
    uint64_t inputs=0, samples=0;
    int64_t lastFlush=0;
    template<class T> static void Values(std::ostream& out,const T* p,int n){for(int i=0;i<n;i++)out<<','<<p[i];}
public:
    static int64_t Now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
    bool active=false;
    bool failed=false;
    std::string path()const{return directory.string();}
    void Start(const std::string& root,const mjModel* model,const std::string& metadata="recording_metadata.json",const std::string& policyFile=""){
        if(active)return;
        failed=false;inputs=samples=0;
        auto stamp=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        directory=std::filesystem::path(root)/"recordings"/("episode-"+std::to_string(stamp));
        try{
            std::filesystem::create_directories(directory.parent_path());
            if(!std::filesystem::create_directory(directory))throw std::runtime_error("Episode already exists");
            std::filesystem::copy_file(std::filesystem::path(root)/metadata,directory/"manifest.json");
            if(!policyFile.empty())std::filesystem::copy_file(std::filesystem::path(root)/policyFile,directory/"policy.onnx");
            for(auto* f:{&input,&frames,&events,&body,&mimic})f->exceptions(std::ios::badbit|std::ios::failbit);
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
            active=true;Event("start",0);Flush();
        }catch(...){Abort();throw;}
    }
    void Event(const char* event,uint64_t sequence){if(active)events<<Now()<<','<<sequence<<','<<event<<'\n';}
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
    }
    void Frame(const mjModel* m,const mjData* d,int steps,const TrackingFrame& f,bool reset,bool calibrated,
               bool apply,bool valid,float left,float right,int mode,double error,bool limited,const std::array<float,29>& reference){
        if(!active)return;
        frames<<Now()<<','<<f.sequence<<','<<steps<<','<<d->time<<','<<reset<<','<<calibrated<<','<<apply<<','<<valid<<','<<left<<','<<right<<','<<mode<<','<<error<<','<<limited;
        Values(frames,reference.data(),29);Values(frames,d->qpos,m->nq);Values(frames,d->qvel,m->nv);Values(frames,d->ctrl,m->nu);
        frames<<'\n';samples++;
    }
    void Mimic(const TrackingFrame& f,int step,double time,const std::array<float,35>& value){
        if(!active)return;
        mimic<<f.sequence<<','<<step<<','<<time;Values(mimic,value.data(),35);mimic<<'\n';
    }
    void Flush(){if(active){input.flush();frames.flush();events.flush();body.flush();mimic.flush();lastFlush=Now();}}
    void Tick(){if(active && Now()-lastFlush>1000000000LL)Flush();}
    void Stop(const char* reason="user_stop"){
        if(!active)return;
        try{
            Event(reason,0);Flush();input.close();frames.close();events.close();body.close();mimic.close();
            std::ofstream done;done.exceptions(std::ios::badbit|std::ios::failbit);
            done.open(directory/"complete.tmp");
            done<<"{\"schema_version\":1,\"reason\":\""<<reason<<"\",\"input_rows\":"<<inputs<<",\"frame_rows\":"<<samples<<"}\n";
            done.close();std::filesystem::rename(directory/"complete.tmp",directory/"complete.json");active=false;
        }catch(...){Abort();throw;}
    }
    void Abort()noexcept{
        active=false;failed=true;
        for(auto* f:{&input,&frames,&events,&body,&mimic}){f->exceptions(std::ios::goodbit);if(f->is_open())f->close();f->clear();}
    }
};

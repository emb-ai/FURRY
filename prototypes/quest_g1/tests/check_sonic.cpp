#include "simulation.h"
#include "policy_catalog.h"
#include "recording.h"
#include <fstream>
#include <iostream>
#include <thread>
static void Check(bool b,const char* why){if(!b)throw std::runtime_error(why);}
int main(int argc,char** argv){try{
    if(argc<3)return 2;std::string assets=argv[1],out=argv[2];auto catalog=questpolicy::Load(assets);
    auto find=[&](const char* id){for(auto& e:catalog)if(e.id==id)return e;throw std::runtime_error("Missing mode");};
    auto entry=find("sonic-vr3");auto policy=questpolicy::VerifiedPolicy(assets,entry);
    Simulation sim(assets,0,"stand",policy,50);
    std::ofstream csv(out+"/native-sonic.csv");csv<<std::setprecision(17);
    std::array<float,35> cmd{};cmd[2]=.793;
    for(int k=0;k<29;k++)cmd[6+k]=sonic::home[k];cmd[24]=cmd[31]=1.2;
    for(int frame=0;frame<100;frame++){
        if(frame==50)cmd[0]=.3;sim.SetWholeBodyReference(cmd);for(int i=0;i<20;i++)sim.Step(false);
        for(int k=0;k<sim.model->nq;k++)csv<<(k?",":"")<<sim.data->qpos[k];csv<<'\n';
    }csv.close();
    auto saved=sim.data->qpos[2];auto bad=policy;bad.plannerPath=assets+"/missing-planner.onnx";
    bool rejected=false;try{sim.SelectPolicy(bad);}catch(...){rejected=true;}
    Check(rejected&&sim.data->qpos[2]==saved&&sim.Backend()==questpolicy::Sonic,"Failed selection changed active policy");
    sim.Reset();Simulation fresh(assets,0,"stand",policy,50);
    for(int i=0;i<200;i++){sim.Step(false);fresh.Step(false);}
    for(int k=0;k<sim.model->nq;k++)Check(std::abs(sim.data->qpos[k]-fresh.data->qpos[k])<1e-10,"SONIC reset leaves stale history/plan");
    sim.Reset();fresh.Reset();
    for(int frame=0;frame<20;frame++){
        auto other=cmd;for(int k=0;k<12;k++)other[6+k]+=.2f*std::sin(k+frame);
        sim.SetWholeBodyReference(cmd);fresh.SetWholeBodyReference(other);
        for(int i=0;i<20;i++){sim.Step(false);fresh.Step(false);}
        for(int k=0;k<sim.model->nq;k++)Check(std::abs(sim.data->qpos[k]-fresh.data->qpos[k])<1e-10,"Virtual legs leaked into native SONIC");
    }
    // Actual catalogue + manifest + complete model identities through recorder/replay.
    auto recordingRoot=std::filesystem::path(out)/"recording-fixture";
    std::filesystem::create_directories(recordingRoot);
    for(const auto& file:std::filesystem::directory_iterator(assets))if(file.is_regular_file()){
        auto link=recordingRoot/file.path().filename();if(!std::filesystem::exists(link))std::filesystem::create_symlink(file.path(),link);
    }
    EpisodeRecorder recorder;recorder.Start(recordingRoot.string(),sim.model,questpolicy::Metadata("stand",entry),entry.file,false,false,entry.secondaryFile);
    auto episode=recorder.path();recorder.Stop();auto manifest=episodereplay::LoadManifest(episode);auto replay=manifest.Policy(assets,episode);
    Check(replay.backend==questpolicy::Sonic&&!replay.plannerPath.empty(),"SONIC recording contract lost");
    std::ofstream(out+"/episode-path.txt")<<episode;
    for(auto id:{"homie-v2","twist2-20k","sonic-vr3"}){
        auto p=questpolicy::VerifiedPolicy(assets,find(id));sim.SelectPolicy(p);
        for(int i=0;i<200;i++)sim.Step(false);Check(sim.data->qpos[2]>.35,"Mode switch fell");
    }
    for(auto name:{"cup","push_t"}){Simulation task(assets,0,name,policy,50);for(int i=0;i<500;i++)task.Step(false);}
    // Android scheduling: separate planner, reset with in-flight work, fresh startup.
    policy.asyncPlanner=true;sim.SelectPolicy(policy);
    for(int frame=0;frame<65;frame++){
        cmd[0]=frame<25?0:.3;sim.SetWholeBodyReference(cmd);for(int i=0;i<20;i++)sim.Step(false);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if(frame==31)sim.Reset();
    }
    std::cout<<"SONIC 2s rollout, transactional load, reset, mode switches, recording hashes and async planner passed; planner_ms="<<sim.PlannerMs()<<"\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

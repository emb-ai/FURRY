#include "guided_capture.h"
#include "recording.h"
#include <iostream>

// Exercise the real recorder with a frozen model and a saved valid body fixture.
int main(int argc,char**argv){
    if(argc!=4)return 2;
    try{
        char error[1024];mjModel* m=mj_loadXML((std::string(argv[1])+"/scene.xml").c_str(),nullptr,error,sizeof(error));
        if(!m)throw std::runtime_error(error);mjData* d=mj_makeData(m);mj_forward(m,d);
        TrackingFrame f;std::ifstream fixture(argv[3]);
        auto pose=[&](TrackedPose& p){for(double& v:p.position)fixture>>v;for(double& v:p.quaternion)fixture>>v;};
        pose(f.head);pose(f.hands[0]);pose(f.hands[1]);
        for(int j=0;j<14;j++){pose(f.body.joints[j]);pose(f.body.rest[j]);f.body.flags[j]=15;}
        if(!fixture)throw std::runtime_error("Invalid body fixture");
        f.location_flags={15,15,15};f.hand_active={true,true};f.body.supported=true;f.body.confidence=1;f.body.skeleton_version=1;
        EpisodeRecorder r;r.Start(argv[2],m);GuidedCapture g;g.Cycle();g.Start(0);g.WritePlan(r.path());
        std::ofstream timeline(r.path()+"/capture_timeline.csv");timeline<<std::setprecision(17)
            <<"receive_ns,sequence,stage,status,focused,source_valid,calibrated,accepted_seconds,stage_seconds\n";
        std::array<float,29> reference{};std::array<float,35> mimic{};mimic[2]=d->qpos[2];
        for(int n=0;n<1100;n++){
            f.sequence=n+1;f.xr_time_ns=f.body.time_ns=1000000000LL+n*10000000LL;
            f.valid=f.body.valid=!(n>=600 && n<700);r.Input(f,0,0,EpisodeRecorder::Now());
            if(n==5){g.Calibrate();r.Event("calibrate",f.sequence);}
            if(n==750 || n==800)g.Pause();
            g.Tick(n*.01,f.body.valid);
            timeline<<EpisodeRecorder::Now()<<','<<f.sequence<<','<<g.stage<<','<<g.Status(true,f.body.valid)<<",1,"<<f.body.valid<<','<<g.calibrated<<','<<g.Total()<<','<<g.accepted[g.stage]<<'\n';
            r.Frame(m,d,0,f,false,n==5,false,f.body.valid,0,0,0,0,false,reference);r.Mimic(f,0,0,mimic);
        }
        timeline.close();g.WriteSummary(r.path());g.Stop();r.Stop();
        std::cout<<r.path()<<'\n';mj_deleteData(d);mj_deleteModel(m);
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}

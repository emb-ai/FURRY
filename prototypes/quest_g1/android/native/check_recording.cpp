#include "recording.h"
#include "simulation.h"
#include <iostream>
#include <cmath>
int main(int argc,char** argv){
    if(argc!=3)return 2;
    try{
        Simulation sim(argv[1]);ArmRetargeter retarget(sim.model);EpisodeRecorder recorder;
        recorder.Start(argv[2],sim.model);
        TrackingFrame f;f.valid=true;f.location_flags={15,15,15};f.hand_active={true,true};
        f.head.position={0,1.65,0};f.hands[0].position={-.25,1.2,-.35};f.hands[1].position={.25,1.2,-.35};
        auto origin=f;std::array<float,29> reference{};
        for(int n=0;n<500;n++){
            bool reset=n==300,calibrate=n==10 || n==310;
            if(reset){sim.Reset();retarget.calibrated=false;recorder.Event("reset",n);}
            f=origin;f.sequence=n+1;f.xr_time_ns=1000000000LL+n*10000000LL;
            f.hands[0].position[1]+=.06*std::sin(n*.02);
            f.valid=!(n>=200 && n<220);if(!f.valid)f.location_flags[1]=0;
            recorder.Input(f,.2,0,EpisodeRecorder::Now());
            if(calibrate){retarget.Calibrate(sim.data,f);recorder.Event("calibrate",f.sequence);}
            bool apply=f.valid && retarget.calibrated && !(n>=100 && n<130);
            if(apply){reference=retarget.Solve(f);sim.SetArmReference(reference);}
            recorder.Frame(sim.model,sim.data,sim.steps,f,reset,calibrate,apply,f.valid,.2,0,1,retarget.error_m,retarget.limited,reference);
            for(int i=0;i<10;i++)sim.Step(false,.2,0);
        }
        recorder.Stop();std::cout<<recorder.path()<<'\n';
        // Interrupted recordings must retain partial streams without a completion marker.
        EpisodeRecorder interrupted;interrupted.Start(argv[2],sim.model);auto partial=interrupted.path();interrupted.Abort();
        if(std::filesystem::exists(std::filesystem::path(partial)/"complete.json"))return 1;
        std::filesystem::remove_all(partial);
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

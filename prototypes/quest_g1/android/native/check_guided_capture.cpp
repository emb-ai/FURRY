#include "guided_capture.h"
#include <iostream>
#include <stdexcept>

static void Check(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
int main(int argc,char**argv){
    try{
        GuidedCapture g;g.Cycle();Check(g.mode==1 && g.Count()==15,"Train plan missing");
        double now=0;g.Start(now);g.Cycle();Check(g.mode==1,"Mode changed during recording");
        auto tick=[&](int n,bool valid=true){for(int i=0;i<n;i++){now+=.01;g.Tick(now,valid);}};
        tick(1000);Check(g.Total()==0,"Uncalibrated input counted");
        g.Calibrate();tick(1000);Check(g.Total()>4.9 && g.Total()<5.1,"Preparation counted as walking");
        double before=g.Total();g.Pause();tick(1000);Check(g.Total()==before,"User pause counted");
        g.Pause();tick(1000,false);Check(g.Total()==before,"Invalid tracking counted");
        tick(1);Check(g.Total()==before,"Invalid interval credited on recovery");
        tick(100);before=g.Total();now+=30;g.Tick(now,true);Check(g.Total()==before,"Long app stall counted");
        g.Invalidate();tick(1000);Check(g.Total()==before,"Recenter kept calibration valid");
        g.Calibrate();tick(100);Check(g.Total()==before,"Recalibration preparation counted");
        while(!g.completed && now<2000)tick(1);
        Check(g.completed && g.Total()==900 && g.stage==14,"Training protocol did not finish exactly 900 seconds");
        for(int i=0;i<15;i++)Check(g.accepted[i]==60,"Unequal phase durations");
        if(argc==2){g.WritePlan(argv[1]);g.WriteSummary(argv[1]);}
        g.Stop();g.Cycle();Check(g.mode==2 && g.Count()==5,"Test plan missing");g.Start(now);g.Calibrate();
        while(!g.completed && now<2500)tick(1);
        Check(g.completed && g.Total()==300,"Test protocol did not finish exactly 300 seconds");
        g.Stop();g.Start(now);g.Calibrate();tick(800);g.Stop();
        Check(!g.completed && g.Total()>2.9 && g.Total()<3.1,"Partial recording marked complete");
        g.Cycle();Check(g.mode==0,"Manual recording inaccessible");
        std::cout<<"Guided capture: full 15/5 minute protocols; pause, tracking loss, recenter, stalls and partial stop passed\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

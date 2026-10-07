#include "simulation.h"
#include "retarget.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

int main(int argc,char** argv){
    if(argc<2 || argc>3)return 2;
    int failures=0;
    int first=argc==3?std::stoi(argv[2]):0,last=argc==3?first+1:7;
    for(int scenario=first;scenario<last;scenario++){
        Simulation sim(argv[1]);
        for(int i=0;i<2000;i++)sim.Step(false);
        ArmRetargeter ik(sim.model);
        TrackingFrame neutral;neutral.valid=true;neutral.head.position={0,1.65,0};
        neutral.hands[0].position={-.25,1.2,-.35};neutral.hands[1].position={.25,1.2,-.35};
        ik.Calibrate(sim.data,neutral);
        double minHeight=1,maxTilt=0,maxError=0,minContact=0,maxDrift=0,maxSpeed=0;
        bool failed=false;
        for(int frame=0;frame<(scenario==6?6000:2000);frame++){
            auto input=neutral;double wave=.5*(1-std::cos(2*3.141592653589793*frame/600.));
            for(int hand=0;hand<2;hand++){
                if(scenario==1)input.hands[hand].position[1]+=.30*wave;
                if(scenario==2)input.hands[hand].position[2]-=.40*wave;
                if(scenario==3)input.hands[hand].position[0]+=(hand==0?1:-1)*.40*wave;
                if(scenario==4)input.hands[hand].quaternion={std::cos(1.5*wave),std::sin(1.5*wave),0,0};
                if(scenario==5){input.hands[hand].position[1]+=.35*wave;input.hands[hand].position[2]-=.25*wave;input.hands[hand].quaternion={std::cos(wave),0,0,std::sin(wave)};}
                if(scenario==6){input.hands[hand].position[1]+=(frame/100)%2?.4:-.3;input.hands[hand].position[0]+=(hand==0?1:-1)*.3;}
            }
            auto reference=ik.Solve(input);maxError=std::max(maxError,ik.error_m);
            sim.SetArmReference(reference);
            try{for(int k=0;k<10;k++)sim.Step(false,scenario%2,scenario%2);}
            catch(const std::exception& e){failed=true;std::printf("FAIL scenario=%d t=%.2f %s\n",scenario,sim.data->time,e.what());break;}
            maxDrift=std::max(maxDrift,std::hypot(sim.data->qpos[0],sim.data->qpos[1]));
            maxSpeed=std::max(maxSpeed,std::hypot(sim.data->qvel[0],sim.data->qvel[1]));
            minHeight=std::min(minHeight,sim.data->qpos[2]);
            double qx=sim.data->qpos[4],qy=sim.data->qpos[5];maxTilt=std::max(maxTilt,std::acos(std::clamp(1-2*(qx*qx+qy*qy),-1.,1.))*180/3.141592653589793);
            for(int c=0;c<sim.data->ncon;c++)minContact=std::min(minContact,sim.data->contact[c].dist);
        }
        double drift=std::hypot(sim.data->qpos[0],sim.data->qpos[1]);
        failed=failed || minHeight<.70 || maxTilt>15 || maxDrift>.30;
        failures+=failed;
        std::printf("scenario=%d result=%s time=%.2f min_height=%.3f tilt=%.1f IK_error=%.3f penetration=%.4f drift=%.3f max_drift=%.3f max_speed=%.3f\n",scenario,failed?"FAIL":"PASS",sim.data->time,minHeight,maxTilt,maxError,minContact,drift,maxDrift,maxSpeed);
        std::fflush(stdout);
    }
    return failures?1:0;
}

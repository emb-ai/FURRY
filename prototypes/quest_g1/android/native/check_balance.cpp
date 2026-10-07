#include "simulation.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

// Physical pushes exercise braking independently of VR and inverse kinematics.
int main(int argc,char**argv){
    if(argc<2)return 2;
    int failures=0;
    for(int test=0;test<6;test++){
        Simulation sim(argv[1]);
        int pelvis=mj_name2id(sim.model,mjOBJ_BODY,"pelvis");
        double maxDrift=0,minHeight=1,maxTilt=0,lateSpeed=0;
        bool failed=false;
        for(int i=0;i<20000;i++){
            // A 24 N s impulse after settling, in each horizontal direction.
            if(test<4 && i>=3000 && i<3300)sim.data->xfrc_applied[6*pelvis+test/2]=(test%2?-1:1)*80.;
            else for(int a=0;a<6;a++)sim.data->xfrc_applied[6*pelvis+a]=0;
            try{sim.Step(test==5);}catch(const std::exception&e){failed=true;std::printf("%s\n",e.what());break;}
            maxDrift=std::max(maxDrift,std::hypot(sim.data->qpos[0],sim.data->qpos[1]));
            minHeight=std::min(minHeight,sim.data->qpos[2]);
            double qx=sim.data->qpos[4],qy=sim.data->qpos[5];maxTilt=std::max(maxTilt,std::acos(std::clamp(1-2*(qx*qx+qy*qy),-1.,1.))*180/3.141592653589793);
            if(i>=18000)lateSpeed+=std::hypot(sim.data->qvel[0],sim.data->qvel[1])/2000;
        }
        double drift=std::hypot(sim.data->qpos[0],sim.data->qpos[1]);
        failed=failed || minHeight<.70 || maxTilt>15 || maxDrift>.30 || drift>.10 || lateSpeed>.10;
        failures+=failed;
        std::printf("test=%d result=%s min_height=%.3f max_tilt=%.2f max_drift=%.3f final_drift=%.3f late_speed=%.3f\n",test,failed?"FAIL":"PASS",minHeight,maxTilt,maxDrift,drift,lateSpeed);std::fflush(stdout);
    }
    return failures?1:0;
}

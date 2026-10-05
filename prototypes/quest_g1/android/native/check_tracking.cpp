#include "simulation.h"
#include "retarget.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

int main(int argc,char** argv){
    if(argc!=2)return 2;
    Simulation sim(argv[1]);
    for(int i=0;i<2000;i++)sim.Step(false);
    ArmRetargeter ik(sim.model);
    TrackingFrame neutral;
    neutral.valid=true;neutral.head.position={0,1.65,0};
    neutral.hands[0].position={-.25,1.2,-.35};neutral.hands[1].position={.25,1.2,-.35};
    ik.Calibrate(sim.data,neutral);
    auto start=ik.Solve(neutral);
    int wrist=mj_name2id(sim.model,mjOBJ_BODY,"left_wrist_yaw_link");
    std::array<double,3> wristStart;std::copy_n(sim.data->xpos+3*wrist,3,wristStart.begin());
    double actualMovement=0;
    double minHeight=1,maxError=0,maxMovement=0,maxRightMovement=0;
    // Move only the left controller; the independent right arm should hold its target.
    for(int frame=0;frame<3000;frame++){
        TrackingFrame input=neutral;
        double wave=.5*(1-std::cos(2*3.141592653589793*frame/600.));
        input.hands[0].position[1]+=.08*wave;
        input.hands[0].position[2]-=.08*wave;
        auto reference=ik.Solve(input);
        maxError=std::max(maxError,ik.error_m);
        for(int j=15;j<22;j++)maxMovement=std::max(maxMovement,std::abs(double(reference[j]-start[j])));
        for(int j=22;j<29;j++)maxRightMovement=std::max(maxRightMovement,std::abs(double(reference[j]-start[j])));
        sim.SetArmReference(reference);
        for(int i=0;i<10;i++)sim.Step(false,0,1);
        minHeight=std::min(minHeight,sim.data->qpos[2]);
        double displacement[3];mju_sub3(displacement,sim.data->xpos+3*wrist,wristStart.data());
        actualMovement=std::max(actualMovement,mju_norm3(displacement));
    }
    double drift=std::hypot(sim.data->qpos[0],sim.data->qpos[1]);
    std::printf("Tracking regression: sim=%.1fs min_height=%.3f drift=%.3f IK_error=%.4f left_motion=%.3f right_motion=%.5f actual_wrist_motion=%.3fm\n",sim.data->time,minHeight,drift,maxError,maxMovement,maxRightMovement,actualMovement);
    return minHeight>.7 && drift<.3 && maxError<.045 && maxMovement>.1 && maxRightMovement<.01 && actualMovement>.04 ? 0:1;
}

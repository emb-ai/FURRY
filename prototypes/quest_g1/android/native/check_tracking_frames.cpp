#include "simulation.h"
#include "retarget.h"
#include <algorithm>
#include <cmath>
#include <iostream>

// No headset needed: axes, rigid-world equivariance, and HMD-only motion.
int main(int argc,char** argv){
    if(argc!=2)return 2;
    Simulation sim(argv[1]);for(int i=0;i<2000;i++)sim.Step(false);
    TrackingFrame origin;origin.valid=true;origin.head.position={0,1.65,0};
    origin.hands[0].position={-.25,1.2,-.35};origin.hands[1].position={.25,1.2,-.35};
    ArmRetargeter still(sim.model),headOnly(sim.model);still.Calibrate(sim.data,origin);headOnly.Calibrate(sim.data,origin);
    double maxHeadError=0,maxFrameError=0;
    for(int frame=0;frame<200;frame++){
        auto moved=origin;double u=frame/199.;
        moved.head.position[0]+=.1*u;moved.head.position[1]+=.1*u;moved.head.position[2]-=.1*u;
        moved.head.quaternion={std::cos(.4*u),0,std::sin(.4*u),0};
        auto a=still.Solve(origin),b=headOnly.Solve(moved);
        for(int j=0;j<29;j++)maxHeadError=std::max(maxHeadError,std::abs(double(a[j]-b[j])));
    }
    for(double yaw:{-1.5707963267948966,1.5707963267948966,3.141592653589793}){
        double q[4]={std::cos(yaw/2),0,std::sin(yaw/2),0},r[9];mju_quat2Mat(r,q);
        auto transform=[&](TrackingFrame f){for(auto* p:{&f.head,&f.hands[0],&f.hands[1]}){
            double xyz[3],quat[4];mju_mulMatVec(xyz,r,p->position.data(),3,3);mju_mulQuat(quat,q,p->quaternion.data());
            for(int k=0;k<3;k++)p->position[k]=xyz[k]+(k==0?2:(k==2?-3:0));
            std::copy_n(quat,4,p->quaternion.begin());
        }return f;};
        ArmRetargeter a(sim.model),b(sim.model);a.Calibrate(sim.data,origin);b.Calibrate(sim.data,transform(origin));
        for(int frame=0;frame<200;frame++){
            auto input=origin;double u=frame/199.;input.hands[0].position[1]+=.08*u;input.hands[0].position[2]-=.08*u;
            input.hands[0].quaternion={std::cos(.2*u),0,std::sin(.2*u),0};
            auto qa=a.Solve(input),qb=b.Solve(transform(input));
            for(int j=0;j<29;j++)maxFrameError=std::max(maxFrameError,std::abs(double(qa[j]-qb[j])));
        }
    }
    std::cout<<"Head-only reference error="<<maxHeadError<<" rad; global-frame equivariance error="<<maxFrameError<<" rad\n";
    return maxHeadError<1e-6 && maxFrameError<1e-5?0:1;
}

// Offline accuracy audit: separates requested wrist targets, IK and policy motion.
#include "simulation.h"
#include "retarget.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

constexpr double pi=3.141592653589793;
double Dist(const double* a,const double* b){double d[3];mju_sub3(d,a,b);return mju_norm3(d);}
double Angle(const double* a,const double* b){double r[9];mju_mulMatMatT(r,a,b,3,3,3);return std::acos(std::clamp((r[0]+r[4]+r[8]-1)*.5,-1.,1.))*180/pi;}
int main(int argc,char** argv){
    if(argc!=2)return 2;
    std::puts("case,neutral_x,neutral_y,neutral_z,request_dx,request_dy,request_dz,ik_dx,ik_dy,ik_dz,ik_target_error_m,policy_wrist_error_m,orientation_error_deg,limited_fraction,max_other_arm_change_rad,min_height");
    for(const std::string name:{"neutral","forward10","up10","out10","down10","forward30","up30","in30","head_up10","rot60","rot120","rot180","rot_jump180","height150","height195","calibration_low","calibration_high"}){
        Simulation sim(argv[1]);for(int i=0;i<2000;i++)sim.Step(false);
        ArmRetargeter solver(sim.model);
        TrackingFrame neutral;neutral.valid=true;neutral.head.position={0,1.65,0};
        neutral.hands[0].position={-.25,1.2,-.35};neutral.hands[1].position={.25,1.2,-.35};
        if(name=="height150" || name=="height195"){
            double scale=(name=="height150"?1.50:1.95)/1.65;
            for(auto* pose:{&neutral.head,&neutral.hands[0],&neutral.hands[1]})for(auto& v:pose->position)v*=scale;
        }
        if(name=="calibration_low")for(auto& hand:neutral.hands)hand.position[1]-=.35;
        if(name=="calibration_high")for(auto& hand:neutral.hands)hand.position[1]+=.35;
        solver.Calibrate(sim.data,neutral);
        mjData* fk=mj_makeData(sim.model);mj_copyData(fk,sim.model,sim.data);
        fk->qpos[0]=fk->qpos[1]=0;fk->qpos[2]=.793;fk->qpos[3]=1;fk->qpos[4]=fk->qpos[5]=fk->qpos[6]=0;mj_forward(sim.model,fk);
        int wrist=mj_name2id(sim.model,mjOBJ_BODY,"left_wrist_yaw_link");
        auto first=solver.Solve(neutral);
        // Measure displacement from the calibrated reference, separately from the
        // initial policy stance (calibration may intentionally change the pose).
        for(auto side:{"left","right"}){int j=15+(std::string(side)=="right"?7:0);
            for(auto part:{"shoulder_pitch","shoulder_roll","shoulder_yaw","elbow","wrist_roll","wrist_pitch","wrist_yaw"})
                fk->qpos[sim.model->jnt_qposadr[mj_name2id(sim.model,mjOBJ_JOINT,(std::string(side)+"_"+part+"_joint").c_str())]]=first[j++];
        }mj_forward(sim.model,fk);
        double origin[3],initialR[9],desired[3],desiredR[9];std::copy_n(fk->xpos+3*wrist,3,origin);std::copy_n(fk->xmat+9*wrist,9,initialR);
        std::vector<int> addresses;
        for(auto side:{"left","right"})for(auto part:{"hip_pitch","hip_roll","hip_yaw","knee","ankle_pitch","ankle_roll"})addresses.push_back(sim.model->jnt_qposadr[mj_name2id(sim.model,mjOBJ_JOINT,(std::string(side)+"_"+part+"_joint").c_str())]);
        for(auto part:{"yaw","roll","pitch"})addresses.push_back(sim.model->jnt_qposadr[mj_name2id(sim.model,mjOBJ_JOINT,(std::string("waist_")+part+"_joint").c_str())]);
        for(auto side:{"left","right"})for(auto part:{"shoulder_pitch","shoulder_roll","shoulder_yaw","elbow","wrist_roll","wrist_pitch","wrist_yaw"})addresses.push_back(sim.model->jnt_qposadr[mj_name2id(sim.model,mjOBJ_JOINT,(std::string(side)+"_"+part+"_joint").c_str())]);
        double limited=0,other=0,minHeight=1,policyError=0;
        for(int frame=0;frame<600;frame++){
            auto input=neutral;double u=std::min(1.,frame/200.);u=.5-.5*std::cos(pi*u);
            double angle=0;
            if(name=="forward10")input.hands[0].position[2]-=.10*u;
            if(name=="up10" || name=="height150" || name=="height195" || name=="calibration_low" || name=="calibration_high")input.hands[0].position[1]+=.10*u;
            if(name=="out10")input.hands[0].position[0]-=.10*u;
            if(name=="down10")input.hands[0].position[1]-=.10*u;
            if(name=="forward30")input.hands[0].position[2]-=.30*u;
            if(name=="up30")input.hands[0].position[1]+=.30*u;
            if(name=="in30")input.hands[0].position[0]+=.30*u;
            if(name=="head_up10")input.head.position[1]+=.10*u;
            if(name.rfind("rot",0)==0){angle=(name=="rot_jump180"?180.:std::stod(name.substr(3)))*pi/180*(name=="rot_jump180"?1.:u);input.hands[0].quaternion={std::cos(angle/2),0,std::sin(angle/2),0};}
            double dx=input.hands[0].position[0]-neutral.hands[0].position[0],dy=input.hands[0].position[1]-neutral.hands[0].position[1],dz=input.hands[0].position[2]-neutral.hands[0].position[2];
            // Intended controller-only translation, before any head-induced correction or workspace clipping.
            desired[0]=origin[0]-.8*dz;desired[1]=origin[1]-.8*dx;desired[2]=origin[2]+.8*dy;
            double q[4]={std::cos(angle/2),0,0,std::sin(angle/2)},rot[9];mju_quat2Mat(rot,q);mju_mulMatMat(desiredR,rot,initialR,3,3,3);
            auto ref=solver.Solve(input);limited+=solver.limited;
            for(int j=22;j<29;j++)other=std::max(other,std::abs(double(ref[j]-first[j])));
            for(int j=0;j<29;j++)fk->qpos[addresses[j]]=ref[j];mj_forward(sim.model,fk);
            sim.SetArmReference(ref);for(int step=0;step<10;step++)sim.Step(false);
            minHeight=std::min(minHeight,sim.data->qpos[2]);
        }
        // Express achieved wrist in the same upright root-relative frame as the IK reference.
        double actual[3],delta[3],rootR[9];mju_sub3(delta,sim.data->xpos+3*wrist,sim.data->qpos);mju_quat2Mat(rootR,sim.data->qpos+3);mju_mulMatTVec(actual,rootR,delta,3,3);actual[2]+=.793;
        policyError=Dist(actual,fk->xpos+3*wrist);
        std::printf("%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%.4f,%.6f,%.4f\n",name.c_str(),origin[0],origin[1],origin[2],desired[0]-origin[0],desired[1]-origin[1],desired[2]-origin[2],fk->xpos[3*wrist]-origin[0],fk->xpos[3*wrist+1]-origin[1],fk->xpos[3*wrist+2]-origin[2],Dist(desired,fk->xpos+3*wrist),policyError,Angle(desiredR,fk->xmat+9*wrist),limited/600,other,minHeight);
        std::fflush(stdout);mj_deleteData(fk);
    }
}

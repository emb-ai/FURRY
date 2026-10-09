#include "foot_floor.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

int main(int argc,char** argv) {
    try {
        if(argc!=2)return 2;
        std::string assets=argv[1];char error[1024];
        std::unique_ptr<mjModel,decltype(&mj_deleteModel)> physical(mj_loadXML((assets+"/scene.xml").c_str(),nullptr,error,sizeof(error)),mj_deleteModel);
        if(!physical)throw std::runtime_error(error);
        std::unique_ptr<mjData,decltype(&mj_deleteData)> data(mj_makeData(physical.get()),mj_deleteData);
        GmrRetargeter ref(assets);FootFloor floor;floor.Load(physical.get(),ref.model());
        FootFloorGuard guard(physical.get(),ref.model());
        int plane=mj_name2id(physical.get(),mjOBJ_GEOM,"floor");
        double maxError=0,maxLift=0;int lifted=0;
        for(int n=0;n<300;n++) {
            std::array<float,35> command{};
            command[0]=.2;command[1]=-.1;command[5]=.3;
            command[2]=.76+.045*std::sin(n*.37);
            command[3]=.15*std::sin(n*.13);command[4]=.2*std::cos(n*.19);
            for(int side=0;side<2;side++) {
                command[6+6*side]=-.2;command[9+6*side]=.4;
                command[10+6*side]=-.2+.3*std::sin(n*.17+side);
                command[11+6*side]=.15*std::cos(n*.21+side);
            }
            // Use nonzero yaw independently of the guard's yaw-zero FK.
            double roll[4]={std::cos(command[3]/2.),std::sin(command[3]/2.),0,0};
            double pitch[4]={std::cos(command[4]/2.),0,std::sin(command[4]/2.),0};
            double yaw[4]={std::cos(n*.03),0,0,std::sin(n*.03)},temp[4];
            auto set=[&](const std::array<float,35>& c) {
                auto* rd=ref.data();rd->qpos[0]=.3;rd->qpos[1]=-.2;rd->qpos[2]=c[2];
                mju_mulQuat(temp,yaw,pitch);mju_mulQuat(rd->qpos+3,temp,roll);
                for(int k=0;k<29;k++)rd->qpos[7+k]=c[6+k];
                mj_kinematics(ref.model(),rd);
                std::copy_n(rd->qpos,7,data->qpos);
                for(int j=1;j<ref.model()->njnt;j++) {
                    int actual=mj_name2id(physical.get(),mjOBJ_JOINT,mj_id2name(ref.model(),mjOBJ_JOINT,j));
                    data->qpos[physical->jnt_qposadr[actual]]=rd->qpos[ref.model()->jnt_qposadr[j]];
                }
                mj_forward(physical.get(),data.get());
            };
            set(command);auto estimated=floor.Heights(ref);
            std::array<double,2> distance{1,1};
            for(int side=0;side<2;side++) {
                std::string prefix=side?"right":"left";
                for(int pad=1;pad<=3;pad++) {
                    int geom=mj_name2id(physical.get(),mjOBJ_GEOM,(prefix+"_foot"+std::to_string(pad)+"_collision").c_str());
                    // Independent MuJoCo signed plane/pad collision distance.
                    distance[side]=std::min(distance[side],double(mj_geomDistance(physical.get(),data.get(),plane,geom,1,nullptr)));
                }
                maxError=std::max(maxError,std::abs(estimated[side]-distance[side]));
            }
            auto corrected=guard.Correct(command);
            for(int k=0;k<35;k++)if(k!=2 && corrected[k]!=command[k])throw std::runtime_error("Floor correction changed a joint/velocity/tilt");
            if(corrected[2]<command[2])throw std::runtime_error("Floor correction lowered a reference");
            if(std::min(distance[0],distance[1])>=0 && corrected!=command)throw std::runtime_error("Clear pose was changed");
            set(corrected);auto height=floor.Heights(ref);
            if(std::min(height[0],height[1]) < -1e-7)throw std::runtime_error("Corrected pose still penetrates floor");
            maxLift=std::max(maxLift,double(corrected[2]-command[2]));lifted+=corrected[2]>command[2];
            if(guard.Correct(corrected)!=corrected)throw std::runtime_error("Floor correction is not idempotent");
        }
        if(maxError>1e-9)throw std::runtime_error("Foot-floor geometry differs from MuJoCo collision distances");
        if(!lifted)throw std::runtime_error("Fixture never penetrated floor");
        std::cout<<"foot floor: 300 poses, collision error="<<maxError<<" m, corrected="<<lifted<<", max lift="<<maxLift<<" m\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}

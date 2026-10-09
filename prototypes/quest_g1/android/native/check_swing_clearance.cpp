#include "swing_clearance.h"
#include <iostream>

int main(int argc,char** argv){
    try{
        if(argc!=2)return 2;GmrRetargeter ref(argv[1]);char error[1024];
        std::unique_ptr<mjModel,decltype(&mj_deleteModel)> physical(mj_loadXML((std::string(argv[1])+"/scene.xml").c_str(),nullptr,error,sizeof(error)),mj_deleteModel);
        if(!physical)throw std::runtime_error(error);
        FootFloor floor;floor.Load(physical.get(),ref.model());SwingClearance swing;swing.Load(physical.get(),ref.model());int lifted=0;
        for(int n=0;n<200;n++){
            std::vector<TrackedPose> targets(14);
            for(int side=0;side<2;side++){
                auto& f=targets[6+side];f.position={.1,double(side)*.2,.06+.03*std::sin(n*.17+side)};
                double pitch=.7*std::sin(n*.13+side);f.quaternion={std::cos(pitch/2),0,std::sin(pitch/2),0};
            }
            auto before=targets;int side=n%2;auto heights=floor.Heights(targets[6],targets[7]);
            swing.Correct(targets,side==0?.06:-.06,.03);auto after=floor.Heights(targets[6],targets[7]);
            for(int k=0;k<14;k++)for(int a=0;a<3;a++)if(k!=6+side || a!=2){if(targets[k].position[a]!=before[k].position[a])throw std::runtime_error("Swing changed another target component");}
            for(int k=0;k<14;k++)if(targets[k].quaternion!=before[k].quaternion)throw std::runtime_error("Swing changed foot/body rotation");
            if(after[side]+1e-9<heights[side] || std::abs(after[1-side]-heights[1-side])>1e-12)throw std::runtime_error("Swing changed support or lowered sole");
            double lift=targets[6+side].position[2]-before[6+side].position[2];
            if(lift<0 || lift>.06+1e-12)throw std::runtime_error("Swing exceeded vertical lift bound");
            if(.03-heights[side]<=.06 && after[side]<.03-1e-9)throw std::runtime_error("Swing missed reachable clearance");
            lifted+=lift>0;
            for(double separation:{-.012,0.,.012}){auto neutral=before;swing.Correct(neutral,separation,.03);for(int k=0;k<14;k++)if(neutral[k].position!=before[k].position)throw std::runtime_error("Standing asymmetry triggered lift");}
            auto disabled=before;swing.Correct(disabled,.1,0);for(int k=0;k<14;k++)if(disabled[k].position!=before[k].position)throw std::runtime_error("Disabled correction changed targets");
            // The threshold and cubic blend must be continuous at entry.
            auto edge=before;swing.Correct(edge,.012000001,.03);if(edge[6].position[2]-before[6].position[2]>.000001)throw std::runtime_error("Swing entry is discontinuous");
        }
        if(lifted<50)throw std::runtime_error("Swing fixture did not exercise lifts");
        std::cout<<"swing targets: 200 orientations, lifts="<<lifted<<", support/rotations preserved, lift <=60 mm\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}

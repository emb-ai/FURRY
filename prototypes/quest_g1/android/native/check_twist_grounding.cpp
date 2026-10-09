#include "gmr.h"
#include <cmath>
#include <iostream>
#include <limits>

int main(int argc,char**argv){
    try{
        if(argc!=2)return 2;
        GmrRetargeter gmr(argv[1]);
        for(int n=0;n<200;n++){
            std::vector<TrackedPose> targets(14);
            for(int i=0;i<14;i++)targets[i].position={.13*i,.09*i,.4*std::sin(.11*n+i)};
            auto before=targets;
            gmr.GroundFootTargets(targets);
            double shift=.1-std::min(before[6].position[2],before[7].position[2]);
            if(std::abs(std::min(targets[6].position[2],targets[7].position[2])-.1)>1e-12)throw std::runtime_error("Foot origins are not grounded at 0.1 m");
            for(int i=0;i<14;i++){
                if(targets[i].position[0]!=before[i].position[0] || targets[i].position[1]!=before[i].position[1] || targets[i].quaternion!=before[i].quaternion)throw std::runtime_error("Grounding changed XY or orientation");
                if(std::abs(targets[i].position[2]-before[i].position[2]-shift)>1e-12)throw std::runtime_error("Grounding changed relative skeleton height");
            }
        }
        bool rejected=false;
        try{std::vector<TrackedPose> invalid(14);invalid[6].position[2]=std::numeric_limits<double>::quiet_NaN();gmr.GroundFootTargets(invalid);}catch(const std::exception&){rejected=true;}
        if(!rejected)throw std::runtime_error("Grounding accepted a nonfinite foot");
        std::cout<<"TWIST2 grounding: 200 skeletons, lowest foot origin=0.1 m, relative geometry preserved\n";
        return 0;
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}
}

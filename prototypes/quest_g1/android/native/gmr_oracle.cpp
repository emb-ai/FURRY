#include "gmr.h"
#include <fstream>
#include <iostream>
#include <iomanip>
int main(int argc,char**argv){
    if(argc!=4)return 2;
    try{
        GmrRetargeter gmr(argv[1]);std::ifstream f(argv[2]);std::ofstream out(argv[3]);out<<std::setprecision(17);
        int frames,count;double height;f>>frames>>count>>height;
        if(count!=int(gmr.tasks().size()))return 3;
        for(int n=0;n<frames;n++){
            std::vector<TrackedPose> human(count);
            for(auto&p:human){for(auto&v:p.position)f>>v;for(auto&v:p.quaternion)f>>v;}
            if(!f)return 4;
            gmr.SetHumanTargets(human,height,true);gmr.Solve();
            for(int j=0;j<gmr.model()->nq;j++)out<<(j?",":"")<<gmr.data()->qpos[j];out<<'\n';
        }
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}

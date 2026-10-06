#include "retarget.h"
#include <fstream>
#include <sstream>
#include <map>
#include <iostream>
#include <algorithm>
#include <cmath>

static std::vector<double> Row(const std::string& line){
    std::stringstream s(line);std::string item;std::vector<double> out;
    while(std::getline(s,item,',')){double x=std::stod(item);if(!std::isfinite(x))throw std::runtime_error("Nonfinite sample");out.push_back(x);}return out;
}
int main(int argc,char** argv){
    if(argc!=3){std::cerr<<"Usage: g1_replay_human ASSETS EPISODE\n";return 2;}
    try{
        std::ifstream manifest(std::string(argv[2])+"/manifest.json");
        std::string metadata((std::istreambuf_iterator<char>(manifest)),std::istreambuf_iterator<char>());
        if(metadata.find("native_gmr_meta_full_body")!=std::string::npos)
            throw std::runtime_error("This is a GMR recording: use g1_replay_gmr instead of legacy wrist replay");
        char error[1024];auto* m=mj_loadXML((std::string(argv[1])+"/scene.xml").c_str(),nullptr,error,sizeof(error));
        if(!m)throw std::runtime_error(error);
        auto* d=mj_makeData(m);
        ArmRetargeter retarget(m);
        std::ifstream inputs(std::string(argv[2])+"/input.csv"),frames(std::string(argv[2])+"/frames.csv");
        if(!inputs || !frames)throw std::runtime_error("Missing episode streams");
        std::map<uint64_t,TrackingFrame> poses;std::string line;std::getline(inputs,line);
        while(std::getline(inputs,line)){
            auto row=Row(line);if(row.size()!=32)throw std::runtime_error("Invalid input row");
            TrackingFrame f;f.sequence=uint64_t(row[0]);f.xr_time_ns=int64_t(row[1]);f.valid=row[3];
            for(int i=0;i<3;i++)f.location_flags[i]=uint64_t(row[4+i]);
            for(int i=0;i<2;i++)f.hand_active[i]=row[7+i];
            TrackedPose* p[]={&f.head,&f.hands[0],&f.hands[1]};
            for(int i=0;i<3;i++){std::copy_n(row.begin()+11+7*i,3,p[i]->position.begin());std::copy_n(row.begin()+14+7*i,4,p[i]->quaternion.begin());}
            poses[f.sequence]=f;
        }
        int count=0,skipped=0;double maxError=0;std::getline(frames,line);
        std::ofstream result(std::string(argv[2])+"/replayed_reference.csv");
        result<<"sequence";for(int i=0;i<29;i++)result<<",reference_"<<i;result<<'\n';
        while(std::getline(frames,line)){
            auto row=Row(line);if(row.size()!=size_t(42+m->nq+m->nv+m->nu))throw std::runtime_error("Invalid state row");
            if(row[4])retarget.calibrated=false;
            auto it=poses.find(uint64_t(row[1]));if(it==poses.end())throw std::runtime_error("State references missing input sequence");
            if(row[5]){
                std::copy_n(row.begin()+42,m->nq,d->qpos);mj_forward(m,d);retarget.Calibrate(d,it->second);
            }
            if(!row[6])continue;
            if(!retarget.calibrated){skipped++;continue;}
            auto reference=retarget.Solve(it->second);
            result<<it->first;
            for(int i=0;i<29;i++){maxError=std::max(maxError,std::abs(double(reference[i])-row[13+i]));result<<','<<reference[i];}
            result<<'\n';count++;
        }
        std::cout<<"Human-input replay: "<<count<<" frames; skipped before calibration="<<skipped<<"; max reference divergence="<<maxError<<" rad\n";
        mj_deleteData(d); // retarget owns independent mjData; model remains alive until process exit.
        return count>0 && maxError<1e-5?0:1;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}

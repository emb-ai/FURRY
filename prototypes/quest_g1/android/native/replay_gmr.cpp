#include "meta_retarget.h"
#include "simulation.h"
#include <fstream>
#include <sstream>
#include <map>
#include <iostream>
#include <algorithm>
#include <cmath>
static std::vector<double> Row(const std::string& line){std::stringstream s(line);std::string item;std::vector<double> row;while(std::getline(s,item,',')){double v=std::stod(item);if(!std::isfinite(v))throw std::runtime_error("Nonfinite recording");row.push_back(v);}return row;}
int main(int argc,char**argv){
    if(argc!=3)return 2;
    try{
        std::string folder=argv[2];Simulation sim(argv[1]);MetaRetargeter meta(argv[1]);
        std::ifstream inputs(folder+"/input.csv"),bodies(folder+"/body.csv"),frames(folder+"/frames.csv"),commands(folder+"/mimic.csv");
        if(!inputs||!bodies||!frames||!commands)throw std::runtime_error("GMR replay requires input/body/frames/mimic streams");
        std::map<uint64_t,TrackingFrame> poses;std::string line;
        std::getline(inputs,line);while(std::getline(inputs,line)){
            auto r=Row(line);if(r.size()!=32)throw std::runtime_error("Invalid input row");
            TrackingFrame f;f.sequence=r[0];f.xr_time_ns=r[1];f.valid=r[3];
            TrackedPose* p[]={&f.head,&f.hands[0],&f.hands[1]};for(int i=0;i<3;i++){std::copy_n(r.begin()+11+7*i,3,p[i]->position.begin());std::copy_n(r.begin()+14+7*i,4,p[i]->quaternion.begin());}poses[f.sequence]=f;
        }
        std::getline(bodies,line);while(std::getline(bodies,line)){
            auto r=Row(line);if(r.size()!=216 || !poses.count(r[0]))throw std::runtime_error("Invalid body row");auto&b=poses.at(r[0]).body;
            b.time_ns=r[1];b.supported=r[2];b.valid=r[3];b.confidence=r[4];b.skeleton_version=r[5];
            for(int i=0;i<14;i++){int a=6+15*i;b.flags[i]=r[a];for(int k=0;k<2;k++){auto&p=k?b.rest[i]:b.joints[i];std::copy_n(r.begin()+a+1+7*k,3,p.position.begin());std::copy_n(r.begin()+a+4+7*k,4,p.quaternion.begin());}}
        }
        std::vector<int64_t> focusPauses;
        std::ifstream events(folder+"/events.csv");std::getline(events,line);
        while(std::getline(events,line)){
            std::stringstream row(line);std::string time,sequence,event;
            std::getline(row,time,',');std::getline(row,sequence,',');std::getline(row,event);
            if(event=="focus_pause")focusPauses.push_back(std::stoll(time));
        }
        size_t pauseIndex=0;
        std::array<float,35> origin{};double blendStart=0,maxError=0;bool wasApplying=false;int count=0,skipped=0;
        std::getline(frames,line);std::string command;std::getline(commands,command);
        while(std::getline(frames,line)){
            if(!std::getline(commands,command))throw std::runtime_error("Missing mimic frame");auto r=Row(line),c=Row(command);
            if(r.size()!=size_t(42+sim.model->nq+sim.model->nv+sim.model->nu)||c.size()!=38 || c[0]!=r[1]||c[1]!=r[2]||c[2]!=r[3])throw std::runtime_error("Mismatched frame/mimic streams");
            while(pauseIndex<focusPauses.size() && focusPauses[pauseIndex]<=r[0]){wasApplying=false;meta.Pause();pauseIndex++;}
            auto it=poses.find(r[1]);if(it==poses.end())throw std::runtime_error("Missing raw frame");const auto&f=it->second;
            std::copy_n(r.begin()+42,sim.model->nq,sim.data->qpos);mj_forward(sim.model,sim.data);
            if(r[4]){meta.calibrated=false;wasApplying=false;}
            if(r[5])meta.Calibrate(sim.model,sim.data,f);
            if(!r[6]){wasApplying=false;meta.Pause();continue;}
            if(!meta.calibrated){skipped++;continue;}
            if(r[5]||!wasApplying){origin.fill(0);origin[2]=sim.data->qpos[2];blendStart=r[3];auto*gm=meta.solver().model();for(int j=1;j<gm->njnt;j++){int source=mj_name2id(sim.model,mjOBJ_JOINT,mj_id2name(gm,mjOBJ_JOINT,j));origin[gm->jnt_qposadr[j]-1]=sim.data->qpos[sim.model->jnt_qposadr[source]];}}
            auto result=meta.Solve(f);double u=std::clamp((r[3]-blendStart)/.5,0.,1.);
            for(int i=0;i<35;i++){result[i]=origin[i]+u*(result[i]-origin[i]);maxError=std::max(maxError,std::abs(result[i]-c[3+i]));}
            wasApplying=true;count++;
        }
        if(std::getline(commands,command))throw std::runtime_error("Extra mimic rows");
        std::cout<<"GMR replay frames="<<count<<" skipped_before_calibration="<<skipped<<" max_reference_error="<<maxError<<'\n';return count>0 && maxError<1e-5?0:1;
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}
}

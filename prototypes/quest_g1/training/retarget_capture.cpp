// Offline capture adapter: use exactly the native Meta/GMR anatomy scaling.
#include "meta_retarget.h"
#include "simulation.h"
#include "foot_floor.h"
#include <fstream>
#include <iostream>
#include <iomanip>
bool Read(std::istream& in,TrackingFrame& f){
 if(!(in>>f.sequence>>f.xr_time_ns>>f.body.time_ns>>f.body.skeleton_version))return false;
 f.valid=f.body.valid=f.body.supported=true;f.body.confidence=1;f.location_flags[0]=15;
 for(double&v:f.head.position)in>>v;for(double&v:f.head.quaternion)in>>v;
 for(int j=0;j<14;j++){f.body.flags[j]=15;for(auto*p:{&f.body.joints[j],&f.body.rest[j]}){for(double&v:p->position)in>>v;for(double&v:p->quaternion)in>>v;}}
 if(!in)throw std::runtime_error("Truncated source frame");return true;
}
int main(int argc,char**argv){
 if(argc!=4)return 2;
 try{
  Simulation sim(argv[1],0);MetaRetargeter meta(argv[1]);std::ifstream src(argv[2]);std::ofstream out(argv[3]);if(!src||!out)throw std::runtime_error("Cannot open stream");
  TrackingFrame f;if(!Read(src,f))throw std::runtime_error("No calibration frame");meta.Calibrate(sim.model,sim.data,f);
  std::ofstream cal(std::string(argv[3])+".calibration.json");cal<<std::setprecision(17)<<"{\"sequence\":"<<f.sequence<<",\"root_scale\":"<<meta.RootScale()<<",\"arm_scale\":"<<meta.ArmScale()<<"}\n";
  FootFloorGuard guard(sim.model,meta.solver().model());out<<std::setprecision(17);
  out<<"sequence,time_ns,ik_error,floor_lift";for(int k=0;k<36;k++)out<<",q"<<k;out<<'\n';
  while(Read(src,f)){
   auto command=meta.Solve(f);guard.Correct(command);auto*d=meta.solver().data();
   out<<f.sequence<<','<<f.body.time_ns<<','<<meta.error<<','<<guard.lastLift;
   for(int k=0;k<36;k++)out<<','<<(d->qpos[k]+(k==2?guard.lastLift:0));out<<'\n';
  }
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}

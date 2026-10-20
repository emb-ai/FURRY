#include "../rubik-canonical-20261015/snapshot_bridge.cpp"
#include <map>
static std::map<mjData*,std::array<double,14>> finger_targets;
static void finger_callback(const mjModel*m,mjData*d){
 auto it=finger_targets.find(d);if(it==finger_targets.end())return;
 for(int a=0,k=0;a<m->nu;a++){
  const char*n=mj_id2name(m,mjOBJ_ACTUATOR,a);if(!n||std::string(n).find("hand_")==std::string::npos)continue;
  double target=it->second[k++];if(!std::isfinite(target))continue;
  d->ctrl[a]=m->actuator_ctrllimited[a]?std::clamp(target,m->actuator_ctrlrange[2*a],m->actuator_ctrlrange[2*a+1]):target;
 }
}
extern "C" void fingers(void*p,const double* targets){auto*s=static_cast<Simulation*>(p);std::array<double,14>t;std::copy_n(targets,14,t.begin());finger_targets[s->data]=t;mjcb_control=finger_callback;}

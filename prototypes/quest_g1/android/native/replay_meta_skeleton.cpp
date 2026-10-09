#include "meta_retarget.h"
#include "simulation.h"
#include "ablation_metrics.h"
#include <fstream>
#include <sstream>
#include <map>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <stdexcept>
// Offline rollout of a human_skeleton_only Quest episode: the robot never ran
// during capture, so there are no recorded resets or commands to follow.
// Meta body frames drive the current adapter, GMR and policy. Each accepted
// segment is cut into windows of at most 30 s; every window resets the robot,
// calibrates on its first usable frame and blends in over 0.5 s. A fall ends
// its window. Raw participant data is supplied locally and must not be committed.
static std::vector<double> Row(const std::string& line){std::stringstream s(line);std::string item;std::vector<double> row;while(std::getline(s,item,',')){double value=std::stod(item);if(!std::isfinite(value))throw std::runtime_error("Nonfinite recording");row.push_back(value);}return row;}
int Run(int argc,char**argv){
 if(argc!=7){std::cerr<<"Usage: replay_meta_skeleton assets scene episode segments.csv clamp|noclamp output.csv\n"
                        "segments.csv: header, then stage,first_sequence,last_sequence per accepted segment\n";return 2;}
 std::string folder=argv[3],mode=argv[5],line;if(mode!="clamp"&&mode!="noclamp")throw std::runtime_error("Unknown toe mode");
 std::map<uint64_t,TrackingFrame> poses;std::ifstream inputs(folder+"/input.csv"),bodies(folder+"/body.csv"),segments(argv[4]);
 if(!inputs||!bodies||!segments)throw std::runtime_error("Missing episode streams or segments");
 std::getline(inputs,line);while(std::getline(inputs,line)){auto r=Row(line);if(r.size()!=32)throw std::runtime_error("Invalid input row");TrackingFrame f;f.sequence=r[0];f.xr_time_ns=r[1];f.valid=r[3];TrackedPose*p[]={&f.head,&f.hands[0],&f.hands[1]};for(int j=0;j<3;j++){std::copy_n(r.begin()+11+7*j,3,p[j]->position.begin());std::copy_n(r.begin()+14+7*j,4,p[j]->quaternion.begin());}poses[f.sequence]=f;}
 std::getline(bodies,line);while(std::getline(bodies,line)){auto r=Row(line);if(r.size()!=216)throw std::runtime_error("Invalid body row");auto& b=poses.at(r[0]).body;b.time_ns=r[1];b.supported=r[2];b.valid=r[3];b.confidence=r[4];b.skeleton_version=r[5];for(int j=0;j<14;j++){int a=6+15*j;b.flags[j]=r[a];for(int k=0;k<2;k++){auto&p=k?b.rest[j]:b.joints[j];std::copy_n(r.begin()+a+1+7*k,3,p.position.begin());std::copy_n(r.begin()+a+4+7*k,4,p.quaternion.begin());}}}
 Simulation sim(argv[1],2,argv[2]);MetaRetargeter meta(argv[1]);meta.EnableToeClamp(mode=="clamp");
 std::ofstream out(argv[6]);if(!out)throw std::runtime_error("Cannot create output");
 out<<std::setprecision(17)<<"sim_s,stage";for(int k=0;k<35;k++)out<<",cmd_"<<k;AblationHeader(out);out<<"\n";
 std::getline(segments,line);int window=0,falls=0;double seconds=0;
 while(std::getline(segments,line)){
  auto s=Row(line);if(s.size()!=3)throw std::runtime_error("Invalid segment row");
  int stage=s[0];uint64_t first=s[1],last=s[2];
  std::vector<const TrackingFrame*> frames;
  for(auto it=poses.lower_bound(first);it!=poses.end()&&it->first<=last;++it)if(it->second.valid&&it->second.body.valid)frames.push_back(&it->second);
  size_t n=0;
  while(n<frames.size()){
   sim.Reset();meta.calibrated=false;meta.Pause();
   while(n<frames.size()&&!meta.calibrated){try{meta.Calibrate(sim.model,sim.data,*frames[n]);}catch(const std::exception&){n++;}}
   if(n>=frames.size())break;
   window++;
   int64_t start=frames[n]->body.time_ns,end=start+int64_t(30e9);
   std::array<float,35> origin{},cmd{};origin[2]=sim.data->qpos[2];
   auto*gm=meta.solver().model();
   for(int j=1;j<gm->njnt;j++){int id=mj_name2id(sim.model,mjOBJ_JOINT,mj_id2name(gm,mjOBJ_JOINT,j));origin[gm->jnt_qposadr[j]-1]=sim.data->qpos[sim.model->jnt_qposadr[id]];}
   bool fell=false;int tick=0;
   for(;;tick++){
    int64_t now=start+int64_t(tick*1e7);
    if(now>=end||n>=frames.size())break;
    bool fresh=false;
    while(n<frames.size()&&frames[n]->body.time_ns<=now){n++;fresh=true;}
    // A changed Meta skeleton is recalibrated in place, as pressing A would.
    if(fresh){try{cmd=meta.Solve(*frames[n-1]);}catch(const std::exception&){meta.Calibrate(sim.model,sim.data,*frames[n-1]);cmd=meta.Solve(*frames[n-1]);}}
    double u=std::clamp(tick*.01/.5,0.,1.);std::array<float,35> blended;
    for(int k=0;k<35;k++)blended[k]=origin[k]+u*(cmd[k]-origin[k]);
    sim.SetWholeBodyReference(blended);
    // Windows are separated by a 1000 s gap so analysis never joins them.
    out<<window*1000+tick*.01<<','<<stage;for(auto v:blended)out<<','<<v;
    AblationRow(out,sim,meta.solver(),window,true,true,ToeTargetClearance(meta));out<<'\n';
    try{for(int k=0;k<10;k++)sim.Step(false,0,0);}
    catch(const std::exception&e){if(std::string(e.what()).rfind("G1 fell:",0)!=0)throw;fell=true;break;}
   }
   seconds+=tick*.01;
   std::cout<<"window="<<window<<" stage="<<stage<<" seconds="<<tick*.01<<" fell="<<fell<<"\n";
   if(fell){falls++;while(n<frames.size()&&frames[n]->body.time_ns<end)n++;}
  }
 }
 std::cout<<"windows="<<window<<" seconds="<<seconds<<" falls="<<falls<<"\n";
 return window?0:2;
}
int main(int argc,char**argv){try{return Run(argc,argv);}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}

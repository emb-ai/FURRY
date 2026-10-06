#include "meta_retarget.h"
#include "simulation.h"
#include <fstream>
#include <sstream>
#include <map>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <stdexcept>
// Counterfactual rollout from recorded resets. It does not restore hidden policy
// state at an arbitrary frame, nor guarantee bit-identical cross-device dynamics.
// Raw participant data is supplied locally and must not be committed.
static std::vector<double> Row(const std::string& line){std::stringstream s(line);std::string item;std::vector<double> row;while(std::getline(s,item,',')){double value=std::stod(item);if(!std::isfinite(value))throw std::runtime_error("Nonfinite recording");row.push_back(value);}return row;}
int Run(int argc,char**argv){
 if(argc!=5){std::cerr<<"Usage: replay_gmr_dynamics assets episode saved|retarget output.csv\n";return 2;}std::string folder=argv[2],mode=argv[3];if(mode!="saved"&&mode!="retarget")throw std::runtime_error("Unknown replay mode");Simulation sim(argv[1]);MetaRetargeter meta(argv[1]);std::string line;
 std::map<uint64_t,TrackingFrame> poses;std::ifstream inputs(folder+"/input.csv"),bodies(folder+"/body.csv"),frames(folder+"/frames.csv"),commands(folder+"/mimic.csv");
 if(!inputs||!bodies||!frames||!commands)throw std::runtime_error("Missing episode streams");
 std::ifstream events(folder+"/events.csv");while(std::getline(events,line)){if(line.find("focus_")!=std::string::npos||line.find("reference_space_change")!=std::string::npos)throw std::runtime_error("This dynamics replay requires an episode without focus/reference-space events");}
 std::getline(inputs,line);while(std::getline(inputs,line)){auto r=Row(line);if(r.size()!=32)throw std::runtime_error("Invalid input row");TrackingFrame f;f.sequence=r[0];f.xr_time_ns=r[1];f.valid=r[3];TrackedPose*p[]={&f.head,&f.hands[0],&f.hands[1]};for(int j=0;j<3;j++){std::copy_n(r.begin()+11+7*j,3,p[j]->position.begin());std::copy_n(r.begin()+14+7*j,4,p[j]->quaternion.begin());}poses[f.sequence]=f;}
 std::getline(bodies,line);while(std::getline(bodies,line)){auto r=Row(line);if(r.size()!=216)throw std::runtime_error("Invalid body row");auto& b=poses.at(r[0]).body;b.time_ns=r[1];b.supported=r[2];b.valid=r[3];b.confidence=r[4];b.skeleton_version=r[5];for(int j=0;j<14;j++){int a=6+15*j;b.flags[j]=r[a];for(int k=0;k<2;k++){auto&p=k?b.rest[j]:b.joints[j];std::copy_n(r.begin()+a+1+7*k,3,p.position.begin());std::copy_n(r.begin()+a+4+7*k,4,p.quaternion.begin());}}}
 std::ofstream out(argv[4]);if(!out)throw std::runtime_error("Cannot create output");out<<std::setprecision(17)<<"wall_s,sim_s,height,tilt,q_error,cmd_error";for(int k=0;k<35;k++)out<<",cmd_"<<k;out<<"\n";
 bool started=false,fell=false,wasApplying=false;std::array<float,35> origin{};double blendStart=0,firstWall=0,minz=1,maxError=0;int segment=0;
 std::getline(frames,line);std::string command;std::getline(commands,command);
 while(std::getline(frames,line)){if(!std::getline(commands,command))throw std::runtime_error("Missing mimic row");auto r=Row(line),c=Row(command);if(r.size()!=size_t(42+sim.model->nq+sim.model->nv+sim.model->nu)||c.size()!=38||c[0]!=r[1]||c[1]!=r[2]||c[2]!=r[3])throw std::runtime_error("Mismatched frame streams");if(!firstWall)firstWall=r[0];
  if(r[4]){if(started)std::cout<<"segment="<<segment<<" fell="<<fell<<" minz="<<minz<<" max_q_error="<<maxError<<"\n";sim.Reset();meta.calibrated=false;started=true;fell=false;wasApplying=false;minz=1;maxError=0;segment++;}
  if(!started||fell)continue;
  auto& f=poses.at(r[1]);std::array<float,35> cmd{};std::copy_n(c.begin()+3,35,cmd.begin());
  if(mode!="saved"){
   if(r[5])meta.Calibrate(sim.model,sim.data,f);
   if(r[6]&&meta.calibrated){if(r[5]||!wasApplying){origin.fill(0);origin[2]=sim.data->qpos[2];blendStart=r[3];auto*gm=meta.solver().model();for(int j=1;j<gm->njnt;j++){int s=mj_name2id(sim.model,mjOBJ_JOINT,mj_id2name(gm,mjOBJ_JOINT,j));origin[gm->jnt_qposadr[j]-1]=sim.data->qpos[sim.model->jnt_qposadr[s]];}}cmd=meta.Solve(f);double u=std::clamp((r[3]-blendStart)/.5,0.,1.);for(int k=0;k<35;k++)cmd[k]=origin[k]+u*(cmd[k]-origin[k]);wasApplying=true;}
   else{meta.Pause();wasApplying=false;}
  }
  double err=0,cmdErr=0;for(int k=0;k<sim.model->nq;k++)err=std::max(err,std::abs(sim.data->qpos[k]-r[42+k]));for(int k=0;k<35;k++)cmdErr=std::max(cmdErr,std::abs(double(cmd[k])-c[k+3]));maxError=std::max(maxError,err);
  if(r[6])sim.SetWholeBodyReference(cmd);else sim.PauseWholeBodyReference();
  double tilt=std::acos(std::clamp(1-2*(std::pow(sim.data->qpos[4],2)+std::pow(sim.data->qpos[5],2)),-1.,1.))*180/3.141592653589793;
  out<<(r[0]-firstWall)*1e-9<<','<<sim.data->time<<','<<sim.data->qpos[2]<<','<<tilt<<','<<err<<','<<cmdErr;for(auto v:cmd)out<<','<<v;out<<'\n';
  try{for(int k=0;k<10;k++)sim.Step(false,r[8],r[9]);}catch(const std::exception&e){fell=true;std::cout<<"fall wall="<<(r[0]-firstWall)*1e-9<<" sim="<<sim.data->time<<"\n";}minz=std::min(minz,sim.data->qpos[2]);
 }
 if(std::getline(commands,command))throw std::runtime_error("Extra mimic row");
 std::cout<<"segment="<<segment<<" fell="<<fell<<" minz="<<minz<<" max_q_error="<<maxError<<"\n";return started?0:2;
}
int main(int argc,char**argv){try{return Run(argc,argv);}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;}}

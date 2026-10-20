#include "simulation.h"
#include "meta_retarget.h"
#include "policy_catalog.h"
#include "camera_pose_servo.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <thread>
bool Read(std::istream& in,TrackingFrame& f){
 if(!(in>>f.sequence>>f.xr_time_ns>>f.body.time_ns>>f.body.skeleton_version))return false;
 f.valid=f.body.valid=f.body.supported=true;f.body.confidence=1;f.location_flags[0]=15;
 for(double&v:f.head.position)in>>v;for(double&v:f.head.quaternion)in>>v;
 for(int j=0;j<14;j++){f.body.flags[j]=15;for(auto*p:{&f.body.joints[j],&f.body.rest[j]}){for(double&v:p->position)in>>v;for(double&v:p->quaternion)in>>v;}}
 if(!in)throw std::runtime_error("Truncated source frame");return true;
}
int main(int argc,char**argv){try{
 std::string assets=argv[2];
 if(std::string(argv[1])=="refs"){
  Simulation s(assets,0,"stand","",50);MetaRetargeter meta(assets);
  std::string mode=argv[5];meta.EnableVirtualLegLift(mode=="lift");meta.EnableNarrowStance(mode=="narrow");
  std::ifstream in(argv[3]);std::ofstream out(argv[4]);out<<std::setprecision(17);
  TrackingFrame f;Read(in,f);meta.Calibrate(s.model,s.data,f);int n=0;int64_t start=9648582087000LL;
  while(Read(in,f)){
   if(f.body.time_ns>start+20100000000LL)break;
   auto cmd=meta.Solve(f);
   if(f.body.time_ns>=start){out<<f.body.time_ns;for(int k=0;k<36;k++)out<<","<<meta.solver().data()->qpos[k];for(float x:cmd)out<<","<<x;out<<"\n";}
   if(++n%5000==0)std::cerr<<mode<<" refs "<<n<<"\n";
  }
  return 0;
 }
 auto catalog=questpolicy::Load(assets);questpolicy::Entry entry;bool found=false;
 for(auto e:catalog)if(e.id==argv[3]){entry=e;found=true;}if(!found)throw std::runtime_error("Unknown policy");
 auto policy=questpolicy::VerifiedPolicy(assets,entry);Simulation s(assets,0,"stand",policy,50);
 // Common native reset pose, zero velocities and empty policy histories.
 Simulation baseline(assets,0,"stand","",50);
 std::copy_n(baseline.data->qpos,s.model->nq,s.data->qpos);std::fill_n(s.data->qvel,s.model->nv,0);mj_forward(s.model,s.data);
 std::ifstream in(argv[4]);std::ofstream out(argv[5]);out<<std::setprecision(17);
 mjData* goal=mj_makeData(s.model);int camera=mj_name2id(s.model,mjOBJ_CAMERA,"ego");
 GmrRetargeter gm(assets);std::array<int,29> qa;
 for(int j=1;j<gm.model()->njnt;j++){int sj=mj_name2id(s.model,mjOBJ_JOINT,mj_id2name(gm.model(),mjOBJ_JOINT,j));qa[j-1]=s.model->jnt_qposadr[sj];}
 std::array<float,35> origin{};origin[2]=s.data->qpos[2];for(int j=0;j<29;j++)origin[6+j]=s.data->qpos[qa[j]];
 std::array<double,36> q;std::array<float,35> cmd;double t;int count=0;
 auto record=[&](double t){out<<t;for(int j=0;j<s.model->nq;j++)out<<","<<s.data->qpos[j];out<<"\n";};record(0);
 while(in>>t){for(double&x:q)in>>x;for(float&x:cmd)in>>x;if(!in)throw std::runtime_error("Truncated reference");
  goal->qpos[0]=q[0];goal->qpos[1]=q[1];for(int k=2;k<7;k++)goal->qpos[k]=q[k];for(int j=0;j<29;j++)goal->qpos[qa[j]]=q[7+j];mj_forward(s.model,goal);
  TrackedPose target;std::copy_n(goal->cam_xpos+3*camera,3,target.position.begin());mju_mat2Quat(target.quaternion.data(),goal->cam_xmat+9*camera);
  if(s.Backend()!=questpolicy::DexWeave)cmd=ApplyCameraCatchUp(cmd,target,s.model,s.data,camera,true,q.data()+3);
  double u=std::clamp(t/.5,0.,1.);for(int j=0;j<35;j++)cmd[j]=origin[j]+u*(cmd[j]-origin[j]);
  if(s.Backend()==questpolicy::DexWeave)s.SetDexWeaveReference(q.data());s.SetWholeBodyReference(cmd);
  try{for(int k=0;k<10;k++){while(!s.Step(false)){std::this_thread::sleep_for(std::chrono::milliseconds(1));}}}
  catch(const std::exception&e){record(s.data->time);std::cerr<<"STOP "<<s.data->time<<" "<<e.what()<<"\n";return 3;}
  record(s.data->time);if(++count%500==0)std::cerr<<entry.id<<" "<<s.data->time<<"s\n";
 }
 mj_deleteData(goal);return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;}}

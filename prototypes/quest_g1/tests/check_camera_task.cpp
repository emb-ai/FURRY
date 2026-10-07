#include "gmr.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
int main(int argc,char**argv){
 if(argc!=2)return 2;
 GmrRetargeter gmr(argv[1]);auto*m=gmr.model();auto*d=gmr.data();
 std::vector<TrackedPose> targets;
 for(const auto&t:gmr.tasks()){
  TrackedPose p;std::copy_n(d->xpos+3*t.body,3,p.position.begin());mju_mat2Quat(p.quaternion.data(),d->xmat+9*t.body);targets.push_back(p);
 }
 auto origin=gmr.CameraPose(),goal=origin;goal.position[0]+=.08;
 double yaw[4]={std::cos(.075),0,0,std::sin(.075)};mju_mulQuat(goal.quaternion.data(),yaw,origin.quaternion.data());
 auto error=[&](){auto p=gmr.CameraPose();double delta[3],inv[4],rel[4],angle[3];mju_sub3(delta,p.position.data(),goal.position.data());mju_negQuat(inv,p.quaternion.data());mju_mulQuat(rel,inv,goal.quaternion.data());mju_quat2Vel(angle,rel,1);return mju_norm3(delta)+mju_norm3(angle);};
 double before=error();gmr.SetTargets(targets);gmr.SetCameraTarget(goal);gmr.Solve();double after=error(),feet=0;
 for(size_t i=0;i<targets.size();i++)if(gmr.tasks()[i].human.find("Foot")!=std::string::npos){double delta[3];mju_sub3(delta,d->xpos+3*gmr.tasks()[i].body,targets[i].position.data());feet=std::max(feet,mju_norm3(delta));}
 printf("camera SE3 error %.6f -> %.6f; planted foot displacement %.6f m\n",before,after,feet);
 if(!(after<before*.95) || feet>.01)throw std::runtime_error("Camera objective / foot priority regression");
 for(int j=1;j<m->njnt;j++)if(m->jnt_limited[j]){double q=d->qpos[m->jnt_qposadr[j]];if(q<m->jnt_range[2*j]-1e-7 || q>m->jnt_range[2*j+1]+1e-7)throw std::runtime_error("Joint limit");}
 gmr.Reset();if(gmr.HasCameraTarget())throw std::runtime_error("Reset retains camera goal");
 // A distant stationary HMD translates the complete reference, including feet,
 // rather than asking the joints to lean a metre toward it. No drift on repeats.
 gmr.SetTargets(targets);auto distant=origin;distant.position[0]-=1;distant.position[1]+=.4;
 std::array<double,36> initial;std::copy_n(d->qpos,36,initial.begin());
 auto shift=gmr.AlignTargetsToCameraXY(distant);
 for(int j=2;j<36;j++)if(std::abs(d->qpos[j]-initial[j])>1e-12)throw std::runtime_error("Translation changes height/joints");
 for(size_t i=0;i<targets.size();i++)for(int a=0;a<3;a++)if(std::abs(d->xpos[3*gmr.tasks()[i].body+a]-targets[i].position[a]-shift[a])>1e-10)throw std::runtime_error("Nonrigid body translation");
 gmr.SetCameraTarget(distant);gmr.Solve();
 for(int frame=0;frame<100;frame++){
  auto repeated=gmr.AlignTargetsToCameraXY(distant);gmr.Solve();
  if(std::hypot(repeated[0],repeated[1])>1e-7)throw std::runtime_error("Stationary camera reference accumulates drift");
 }
 auto positioned=gmr.CameraPose();double delta[3];mju_sub3(delta,positioned.position.data(),distant.position.data());
 if(mju_norm3(delta)>1e-7)throw std::runtime_error("Reference does not reach distant camera");
 std::puts("Uniform camera translation: floor height, joint angles and stationary reference preserved");

}

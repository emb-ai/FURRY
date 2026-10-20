#include "simulation.h"
#include "episode_manifest.h"
#include <algorithm>
#include <cmath>
#include <iostream>
static thread_local std::string error;
extern "C" {
const char* last_error(){return error.c_str();}
void* create(const char* episode){try{
 auto manifest=episodereplay::LoadManifest(episode);
 auto* s=new Simulation(episode,0,"cup",manifest.Policy(episode,episode),manifest.policyHz);
 manifest.ValidateDimensions(s->model->nq,s->model->nv,s->model->nu);return s;
}catch(const std::exception&e){error=e.what();return nullptr;}}
int reset_random(void*p,unsigned seed){try{static_cast<Simulation*>(p)->NewSceneTrial(seed);return 0;}catch(const std::exception&e){error=e.what();return -1;}}
int random_layout(void*p,unsigned seed){try{auto*s=static_cast<Simulation*>(p);s->collection.Reset(s->model,s->data,seed);return 0;}catch(const std::exception&e){error=e.what();return -1;}}
void destroy(void*p){delete static_cast<Simulation*>(p);}
void restore(void*p,const double*q,const double*v,const double*u,const double*mp,const double*mq){
 auto*s=static_cast<Simulation*>(p);s->Reset();auto*m=s->model;auto*d=s->data;
 std::copy_n(q,m->nq,d->qpos);std::copy_n(v,m->nv,d->qvel);std::copy_n(u,m->nuserdata,d->userdata);
 std::copy_n(mp,3*m->nmocap,d->mocap_pos);std::copy_n(mq,4*m->nmocap,d->mocap_quat);
 d->userdata[CollectionTask::Elapsed]=0;d->userdata[CollectionTask::Settled]=0;d->userdata[CollectionTask::Outcome]=0;
 for(int i=0;i<29;i++)s->target[i]=q[s->qadr[i]];
 for(int i=0;i<14;i++)s->decoupled_command.arms[i]=q[s->qadr[15+i]];
 d->time=0;s->steps=0;mj_forward(m,d);
}
int idle(void*p,int n){auto*s=static_cast<Simulation*>(p);try{for(int i=0;i<n;i++)s->Step(false);return 0;}catch(const std::exception&e){error=e.what();return -1;}}
void teacher_frame(void*p,const double*q,const double*v,const double*u,const double*mp,const double*mq,int phase){
 auto*s=static_cast<Simulation*>(p);auto*m=s->model;auto*d=s->data;
 std::copy_n(q,m->nq,d->qpos);std::copy_n(v,m->nv,d->qvel);std::copy_n(u,m->nuserdata,d->userdata);
 std::copy_n(mp,3*m->nmocap,d->mocap_pos);std::copy_n(mq,4*m->nmocap,d->mocap_quat);
 s->steps=phase;d->time=phase*.001;mj_forward(m,d);
}
void dump_model(void*p,const char*path){auto*s=static_cast<Simulation*>(p);mj_saveModel(s->model,path,nullptr,0);}
void read_state(void*p,double*q,double*v,double*u,double*mp,double*mq){
 auto*s=static_cast<Simulation*>(p);auto*m=s->model;auto*d=s->data;
 std::copy_n(d->qpos,m->nq,q);std::copy_n(d->qvel,m->nv,v);std::copy_n(d->userdata,m->nuserdata,u);
 std::copy_n(d->mocap_pos,3*m->nmocap,mp);std::copy_n(d->mocap_quat,4*m->nmocap,mq);
}
int step(void*p,const float*a,int n){auto*s=static_cast<Simulation*>(p);try{
 std::array<float,35>c;std::copy_n(a,35,c.begin());s->SetWholeBodyReference(c);
 for(int i=0;i<n;i++){
  double t=s->data->time;s->Step(false,std::clamp(double(a[35]),0.,1.),std::clamp(double(a[36]),0.,1.));
  if(std::abs(s->data->time-t-s->model->opt.timestep)>1e-9)throw std::runtime_error("Physics time reset");
  for(auto w:{mjWARN_BADQPOS,mjWARN_BADQVEL,mjWARN_BADQACC,mjWARN_BADCTRL})if(s->data->warning[w].number)throw std::runtime_error("Physics numerical warning");
  if(s->data->userdata[CollectionTask::Outcome])break;
 }
 return int(s->data->userdata[CollectionTask::Outcome]);
}catch(const std::exception&e){error=e.what();return -1;}}
int contacts(void*p){auto*s=static_cast<Simulation*>(p);int cup=mj_name2id(s->model,mjOBJ_BODY,"cup"),n=0;
 for(int i=0;i<s->data->ncon;i++)for(int side=0;side<2;side++){
  auto&c=s->data->contact[i];int a=c.geom[side],b=c.geom[1-side];if(a<0||b<0)continue;
  if(s->model->geom_bodyid[a]==cup&&(s->model->geom_contype[b]&4)&&c.dist<=0)n++;
 }return n;
}
}

#include "simulation.h"
#include "policy_catalog.h"
#include <iostream>
extern "C" {
void* create(const char* assets,const char* scene,double x){try{questpolicy::RuntimePolicy p;for(auto&e:questpolicy::Load(assets))if(e.id=="homie-v2")p=questpolicy::VerifiedPolicy(assets,e);p.standing=true;auto*s=new Simulation(assets,0,scene,p,50);if(x>=0)s->data->qpos[0]=x;mj_forward(s->model,s->data);return s;}catch(const std::exception&e){std::cerr<<e.what();return nullptr;}}
void grip_effort(void*p,double force){auto*s=static_cast<Simulation*>(p);for(int a=0;a<s->model->nu;a++){const char*n=mj_id2name(s->model,mjOBJ_ACTUATOR,a);if(n&&std::string(n).find("hand_")!=std::string::npos){s->model->actuator_forcelimited[a]=1;s->model->actuator_forcerange[2*a]=-force;s->model->actuator_forcerange[2*a+1]=force;}}}
void destroy(void* p){delete static_cast<Simulation*>(p);}
int state(void*p,double*q){auto*s=static_cast<Simulation*>(p);std::copy_n(s->data->qpos,s->model->nq,q);return s->model->nq;}
void correct(void*p,const float*cmd,const double*goals,float*out){auto*s=static_cast<Simulation*>(p);std::array<float,35>c;std::copy_n(cmd,35,c.begin());std::vector<TrackedPose>g(14);for(int h=0;h<2;h++){std::copy_n(goals+7*h,3,g[12+h].position.begin());std::copy_n(goals+7*h+3,4,g[12+h].quaternion.begin());}auto result=s->CorrectStandingArms(c,g);std::copy(result.begin(),result.end(),out);}
int step(void*p,const float*cmd,int n){auto*s=static_cast<Simulation*>(p);std::array<float,35>c;std::copy_n(cmd,35,c.begin());try{s->SetWholeBodyReference(c);for(int i=0;i<n;i++)s->Step(false);return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
void initial_cup(void*p,double x,double y){auto*s=static_cast<Simulation*>(p);if(s->steps!=0)throw std::runtime_error("Initial fixture placement requires reset");int j=mj_name2id(s->model,mjOBJ_JOINT,"cup_free");if(j<0)return;int q=s->model->jnt_qposadr[j];s->data->qpos[q]=x;s->data->qpos[q+1]=y;mj_forward(s->model,s->data);}
void seed(void*p,unsigned int n){static_cast<Simulation*>(p)->NewSceneTrial(n);}
void task(void*p,double*out){auto*s=static_cast<Simulation*>(p);std::copy_n(s->data->userdata,s->model->nuserdata,out);}
int step_grip(void*p,const float*cmd,int n,double l,double r){auto*s=static_cast<Simulation*>(p);std::array<float,35>c;std::copy_n(cmd,35,c.begin());try{s->SetWholeBodyReference(c);for(int i=0;i<n;i++)s->Step(false,l,r);return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
int contacts(void*p,int side){auto*s=static_cast<Simulation*>(p);int object=mj_name2id(s->model,mjOBJ_BODY,"cup"),count=0;for(int c=0;c<s->data->ncon;c++)for(int k=0;k<2;k++){int a=s->data->contact[c].geom[k],b=s->data->contact[c].geom[1-k];if(a<0||b<0||s->model->geom_bodyid[b]!=object)continue;const char*n=mj_id2name(s->model,mjOBJ_BODY,s->model->geom_bodyid[a]);if(n&&std::string(n).rfind(side?"right_hand":"left_hand",0)==0)count++;}return count;}

void diagnostic(void*p){auto*s=static_cast<Simulation*>(p);int object=mj_name2id(s->model,mjOBJ_BODY,"cup");mjtNum v[6];mj_objectVelocity(s->model,s->data,mjOBJ_BODY,object,v,0);std::cout<<"STATE ";for(int i=0;i<s->model->nuserdata;i++)std::cout<<s->data->userdata[i]<<" ";std::cout<<" VELOCITY ";for(auto x:v)std::cout<<x<<" ";std::cout<<" UPRIGHT "<<s->data->xmat[9*object+8]<<"\n";for(int i=0;i<s->data->ncon;i++){auto&c=s->data->contact[i];if(c.geom[0]<0||c.geom[1]<0)continue;if(s->model->geom_bodyid[c.geom[0]]==object||s->model->geom_bodyid[c.geom[1]]==object)std::cout<<mj_id2name(s->model,mjOBJ_GEOM,c.geom[0])<<" / "<<mj_id2name(s->model,mjOBJ_GEOM,c.geom[1])<<" "<<c.dist<<"\n";}}

}

extern "C" {
void* create_policy(const char*assets,const char*scene,const char*id){try{for(auto&e:questpolicy::Load(assets))if(e.id==id)return new Simulation(assets,0,scene,questpolicy::VerifiedPolicy(assets,e),50);throw std::runtime_error("No policy");}catch(const std::exception&e){std::cerr<<e.what();return nullptr;}}
int warnings(void*p){auto*s=static_cast<Simulation*>(p);int n=0;for(auto&w:s->data->warning)n+=w.number;return n;}
int rubik_contacts(void*p,int side){auto*s=static_cast<Simulation*>(p);RubikTask t;t.Initialize(s->model);return t.Contacts(s->model,s->data)[side];}
void velocity(void*p,double*out){auto*s=static_cast<Simulation*>(p);std::copy_n(s->data->qvel,s->model->nv,out);}
void contact_min(void*p,double*out){auto*s=static_cast<Simulation*>(p);out[0]=0;out[1]=s->data->ncon;for(int i=0;i<s->data->ncon;i++)out[0]=std::min(out[0],s->data->contact[i].dist);}
}

extern "C" void tune(void*p,int noslip,double force){auto*s=static_cast<Simulation*>(p);s->model->opt.noslip_iterations=noslip;if(noslip)s->model->opt.enableflags|=mjENBL_MULTICCD;if(force>0)grip_effort(p,force);}
extern "C" int supports(void*p){auto*s=static_cast<Simulation*>(p);int count=0;for(int c=0;c<s->data->ncon;c++){auto&ct=s->data->contact[c];if(ct.geom[0]<0||ct.geom[1]<0)continue;double force[6];mj_contactForce(s->model,s->data,c,force);if(force[0]<.08)continue;for(int side=0;side<2;side++){const char*a=mj_id2name(s->model,mjOBJ_BODY,s->model->geom_bodyid[ct.geom[side]]);const char*b=mj_id2name(s->model,mjOBJ_BODY,s->model->geom_bodyid[ct.geom[1-side]]);if(!a||!b||std::string(a).rfind("rubik_cubie_",0)!=0)continue;std::string other(b);if(other.find("_hand_")==std::string::npos&&other.find("_wrist_yaw_link")==std::string::npos&&other.rfind("rubik_",0)!=0)count++;}}return count;}
extern "C" int step_idle(void*p,int n){auto*s=static_cast<Simulation*>(p);try{for(int i=0;i<n;i++)s->Step(false);return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
extern "C" void reset_idle(void*p){static_cast<Simulation*>(p)->Reset();}

extern "C" int can_switch_face(void*p,int face){auto*s=static_cast<Simulation*>(p);auto*m=mj_copyModel(nullptr,s->model);auto*d=mj_copyData(nullptr,m,s->data);RubikTask t;t.Initialize(m);bool ok=t.Activate(m,d,face);mj_deleteData(d);mj_deleteModel(m);return ok;}

extern "C" void reset_arms(void*p){static_cast<Simulation*>(p)->ResetArmCompensation();}

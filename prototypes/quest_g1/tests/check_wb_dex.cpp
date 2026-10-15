#include "collection_task.h"
#include <iostream>
static void Check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
int main(int argc,char**argv){try{
 Check(argc==2,"assets required");char error[2048];auto path=std::string(argv[1])+"/scene-wb_dex.xml";
 auto*m=mj_loadXML(path.c_str(),nullptr,error,sizeof(error));if(!m)throw std::runtime_error(error);auto*d=mj_makeData(m);CollectionTask task;task.Initialize(m);
 Check(std::abs(task.fallHeight-.25)<1e-12,"Missing crouch-aware fall threshold");
 for(double height:{.77,.50,.499,.40,.35,.26}){
  mj_resetData(m,d);task.Reset(m,d,42);d->qpos[2]=height;mj_forward(m,d);task.Finish(m,d);
  Check(d->userdata[CollectionTask::Outcome]==0,"Upright squat classified as fall");
 }
 mj_resetData(m,d);task.Reset(m,d,42);d->qpos[2]=.24;mj_forward(m,d);task.Finish(m,d);
 Check(d->userdata[CollectionTask::Outcome]==2,"Ground-level pelvis not rejected");
 mj_resetData(m,d);task.Reset(m,d,42);d->qpos[3]=std::cos(.5);d->qpos[4]=std::sin(.5);mj_forward(m,d);task.Finish(m,d);
 Check(d->userdata[CollectionTask::Outcome]==2,"Tipped robot not rejected");
 // A complete placement can score while the operator is still crouched.
 mj_resetData(m,d);task.Reset(m,d,42);d->qpos[0]=-2;d->qpos[2]=.35;
 int q=m->jnt_qposadr[task.joint[0]];d->qpos[q+2]=task.start[0][4]+.1;mj_forward(m,d);task.Finish(m,d);
 std::copy_n(task.goal[0].begin(),3,d->qpos+q);mj_forward(m,d);
 for(int i=0;i<501;i++)task.Finish(m,d);
 Check(d->userdata[CollectionTask::Outcome]==1,"Crouched placement cannot finish");
 // Old episode models retain the original failure definition for replay.
 int numeric=mj_name2id(m,mjOBJ_NUMERIC,"collection_fall");m->names[m->name_numericadr[numeric]]='x';CollectionTask legacy;legacy.Initialize(m);
 Check(legacy.fallHeight==.50,"Legacy fall semantics changed");
 mj_resetData(m,d);legacy.Reset(m,d,42);d->qpos[2]=.4;mj_forward(m,d);legacy.Finish(m,d);
 Check(d->userdata[CollectionTask::Outcome]==2,"Legacy fall replay changed");
 mj_deleteData(d);mj_deleteModel(m);std::cout<<"WB-Dex squat, genuine falls, placement and legacy replay passed\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

#include "rubik_task.h"
#include <iostream>
#include <vector>
static void Require(bool v,const char* message){if(!v)throw std::runtime_error(message);}
struct Fixture {
 mjModel* m; mjData* d; RubikTask task;
 explicit Fixture(const char* path){char error[2048]{};m=mj_loadXML(path,nullptr,error,sizeof(error));if(!m)throw std::runtime_error(error);d=mj_makeData(m);task.Initialize(m);task.Reset(m,d);task.Activate(m,d,4);}
 ~Fixture(){mj_deleteData(d);mj_deleteModel(m);}
 void Step(int count,double left,double right,bool intent,double torque=0){
  for(int step=0;step<count;step++){
   d->userdata[RubikTask::RotationIntent]=intent?1:0;
   std::vector<double> q(d->qpos,d->qpos+m->nq),v(d->qvel,d->qvel+m->nv);
   task.Begin(m,d,left,right);
   for(int k=0;k<m->nq;k++)Require(q[k]==d->qpos[k],"Begin changed qpos");
   for(int k=0;k<m->nv;k++)Require(v[k]==d->qvel[k],"Begin changed velocity");
   mju_zero(d->xfrc_applied,6*m->nbody);
   // One corner receives the contact-load surrogate; the robot acceptance
   // separately uses real finger contacts, never this unit-test load.
   for(int i=0;i<8;i++)if(RubikTask::OnFace(d->userdata+RubikTask::Corners+i*7,4)){
    d->xfrc_applied[6*task.cubie[i]+5]=torque;break;
   }
   mj_step(m,d);task.Finish(m,d);
  }
 }
 double Angle()const{return d->qpos[task.qa[4]];}
};
int main(int argc,char**argv){try{
 Require(argc==3,"Expected guarded and archived-v1 fixtures");
 Fixture guarded(argv[1]);Require(guarded.task.gripGuard,"Guard absent");
 guarded.Step(200,1,1,false);double initial=guarded.Angle();guarded.Step(150,1,.6,false,.02);
 double closed=std::abs(guarded.Angle()-initial)*180/RubikTask::Pi;
 Require(closed<1,"Closing fingers moved the guarded layer");
 guarded.Step(300,1,.6,true);Require(guarded.m->dof_frictionloss[guarded.task.va[4]]==0,"Intent plus settled grip did not release brake");
 initial=guarded.Angle();guarded.Step(150,1,.6,true,.02);double turning=std::abs(guarded.Angle()-initial)*180/RubikTask::Pi;
 Require(turning>30,"Intentional physical rotation was blocked");
 guarded.Step(1,1,.4,true);Require(guarded.m->dof_frictionloss[guarded.task.va[4]]==guarded.task.brakeFriction,"Grip change did not engage brake despite stale intent");
 guarded.Step(50,1,.4,true);auto*snapshot=mj_copyData(nullptr,guarded.m,guarded.d);
 guarded.Step(300,1,.4,true,.01);std::vector<double> expected(guarded.d->qpos,guarded.d->qpos+guarded.m->nq);
 mj_copyData(guarded.d,guarded.m,snapshot);guarded.Step(300,1,.4,true,.01);
 for(int k=0;k<guarded.m->nq;k++)Require(std::abs(guarded.d->qpos[k]-expected[k])<1e-10,"Guard snapshot replay changed dynamics");
 mj_deleteData(snapshot);
 Fixture seating(argv[1]);seating.Step(300,1,1,true);for(int i=0;i<500&&seating.Angle()<89*RubikTask::Pi/180;i++)seating.Step(1,1,1,true,.02);
 Require(seating.d->userdata[RubikTask::Turns]>=1,"No physical quarter before seating");
 seating.Step(5000,1,1,false);
 double seatingError=std::abs(seating.Angle()-seating.d->userdata[RubikTask::Start]);
 Require(seatingError<.035,"Guard prevented magnetic seating after a quarter");
 Require(seating.task.Activate(seating.m,seating.d,0),"Seated layer cannot switch to another face");
 std::cout<<"seating_error_degrees="<<seatingError*180/RubikTask::Pi<<" next_face_activated=true\n";
 guarded.task.Reset(guarded.m,guarded.d);
 Require(guarded.d->userdata[RubikTask::GripLeft]==0&&guarded.d->userdata[RubikTask::GripRight]==0&&guarded.d->userdata[RubikTask::GripSettled]==0,"Reset retained grip history");
 Require(guarded.task.intentHistory.empty()&&!guarded.task.rotationIntent,"Reset retained rotation history");
 for(int f=0;f<6;f++)Require(guarded.m->dof_frictionloss[guarded.task.va[f]]==0,"Reset retained hinge friction");
 Fixture legacy(argv[2]);Require(!legacy.task.gripGuard&&legacy.m->nuserdata==64,"Archived scene not using v1 semantics");
 legacy.Step(200,1,1,false);initial=legacy.Angle();legacy.Step(150,1,.6,false,.02);double old=std::abs(legacy.Angle()-initial)*180/RubikTask::Pi;
 Require(old>30,"Archived free layer unexpectedly guarded");
 std::cout<<"corner_load_degrees legacy="<<old<<" guarded="<<closed<<" intentional="<<turning<<" snapshot_exact=true no_state_teleport=true\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

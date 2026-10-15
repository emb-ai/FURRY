#include "collection_task.h"
#include "tennis_task.h"
#include "recording.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <set>
static void Check(bool ok,const char* reason){if(!ok)throw std::runtime_error(reason);}
int main(int argc,char**argv){try{
 Check(argc==2,"assets required");
 for(const auto* name:{"cup","wb_dex","bimanual","basket"}){
  char error[2048]{};auto path=std::string(argv[1])+"/scene-"+name+".xml";
  mjModel* m=mj_loadXML(path.c_str(),nullptr,error,sizeof error);if(!m)throw std::runtime_error(error);
  mjData* d=mj_makeData(m);CollectionTask task;task.Initialize(m);Check(task.kind>0,"Missing task");std::set<int> locations,targets;int shortTrials=0,longTrials=0;
  for(uint32_t seed=0;seed<128;seed++){
   // Park the uninitialized robot for prop/fixture layout checks.
   // Policy reset poses are exercised by check_standing_control.
   mj_resetData(m,d);d->qpos[0]=-2;task.Reset(m,d,seed);std::vector<double> saved(d->qpos,d->qpos+m->nq);task.Reset(m,d,seed);
   Check(std::equal(saved.begin(),saved.end(),d->qpos),"Seed not repeatable");
   if(task.cupGoalMocap>=0){
    auto target=task.Goal(m,d,0);int q=m->jnt_qposadr[task.joint[0]];
    double length=std::hypot(d->qpos[q]-target[0],d->qpos[q+1]-target[1]);
    Check(target[0]>=.40&&target[0]<=.49&&std::abs(target[1])<=.28,"Goal outside reachable table region");
    Check((length>=.15&&length<=.30)||(length>=.32&&length<=.55),"Bad distance band");
    if(length<=.30)shortTrials++;else longTrials++;
    targets.insert(int(target[1]*10000));
    auto* copy=mj_copyData(nullptr,m,d);auto copied=task.Goal(m,copy,0);
    Check(target==copied,"Copied scene lost goal");mj_deleteData(copy);
    task.Reset(m,d,seed);Check(task.Goal(m,d,0)==target,"Goal seed not repeatable");
    mj_forward(m,d);int marker=mj_name2id(m,mjOBJ_BODY,"cup_goal_marker");
    for(int k=0;k<3;k++)Check(std::abs(d->xpos[3*marker+k]-target[k])<1e-12,"Marker/scoring mismatch");
   }
   for(int i=0;i<task.count;i++){
    int q=m->jnt_qposadr[task.joint[i]];auto b=task.start[i];
    Check(d->qpos[q]>=b[0]&&d->qpos[q]<=b[1]&&(task.basketSides?std::abs(d->qpos[q+1]):d->qpos[q+1])>=b[2]&&(task.basketSides?std::abs(d->qpos[q+1]):d->qpos[q+1])<=b[3],"Object outside start region");
    locations.insert(int(d->qpos[q]*10000));
    for(int c=0;c<d->ncon;c++){
     int a=d->contact[c].geom[0],other=d->contact[c].geom[1];
     if(a>=0&&other>=0&&(m->geom_bodyid[a]==task.body[i]||m->geom_bodyid[other]==task.body[i]))Check(d->contact[c].dist>-.001,(std::string("Initial prop interpenetration scene=")+name+" seed="+std::to_string(seed)+" bodies="+mj_id2name(m,mjOBJ_BODY,m->geom_bodyid[a])+","+mj_id2name(m,mjOBJ_BODY,m->geom_bodyid[other])+" dist="+std::to_string(d->contact[c].dist)).c_str());
    }
   }
   task.Finish(m,d);Check(!d->userdata[CollectionTask::Outcome],"Initial layout already successful");
  }
  Check(locations.size()>100,"Insufficient randomization");
  if(task.cupGoalMocap>=0){Check(targets.size()>100,"Goal not randomized");Check(shortTrials>40&&longTrials>40,"Missing short/long diversity");}
  // Resting at goal without pickup must not count. Then lift, place, release,
  // wait .5 s and confirm success. Task score cannot change physics itself.
  for(int i=0;i<task.count;i++){
   int q=m->jnt_qposadr[task.joint[i]];std::copy_n(task.Goal(m,d,i).begin(),3,d->qpos+q);d->qpos[q+3]=1;std::fill_n(d->qpos+q+4,3,0);
  }
  mj_forward(m,d);
  if(task.kind!=4){for(int k=0;k<600;k++)task.Finish(m,d);Check(!d->userdata[CollectionTask::Outcome],"Success without lift");}
  for(int i=0;i<task.count;i++)d->qpos[m->jnt_qposadr[task.joint[i]]+2]=task.start[i][4]+.10;
  mj_forward(m,d);task.Finish(m,d);
  if(task.requireHands){
   Check(!d->userdata[CollectionTask::Lift0]&&!d->userdata[CollectionTask::Lift1],"Bimanual lift without assigned hands");
   for(int i=0;i<2;i++){
    int obj=-1,hand=-1;
    for(int g=0;g<m->ngeom;g++){
     if(m->geom_bodyid[g]==task.body[i])obj=g;
     const char* n=mj_id2name(m,mjOBJ_BODY,m->geom_bodyid[g]);
     if((m->geom_contype[g]&4)&&n&&std::string(n).rfind(i==0?"left_":"right_",0)==0)hand=g;
    }
    Check(obj>=0&&hand>=0,"Missing hand/object collider");
    mjContact c{};c.geom[0]=obj;c.geom[1]=hand;c.dist=-.0001;
    Check(mj_addContact(m,d,&c)==0,"Cannot add hand contact fixture");
   }
   task.Finish(m,d);Check(d->userdata[CollectionTask::Lift0]&&d->userdata[CollectionTask::Lift1],"Assigned-hand lifts rejected");
  }
  if(task.doorJoint>=0){
   // A placement without passing the doorway must not finish the task.
   std::copy_n(task.goal[0].begin(),3,d->qpos+m->jnt_qposadr[task.joint[0]]);
   mj_forward(m,d);for(int k=0;k<501;k++)task.Finish(m,d);
   Check(!d->userdata[CollectionTask::Outcome],"Basket shortcut skipped door");
   int q=m->jnt_qposadr[task.joint[0]];
   d->qpos[q]=task.doorX+.2;d->qpos[q+1]=0;d->qpos[q+2]=.7;
   d->qpos[0]=task.doorX-.01;d->qpos[1]=0;
   d->qpos[m->jnt_qposadr[task.doorJoint]]=1.3;
   mj_forward(m,d);task.Finish(m,d);
   d->qpos[0]=task.doorX+.01;mj_forward(m,d);task.Finish(m,d);
   Check(d->userdata[CollectionTask::Lift1],"Door passage not recognized");
  }
  for(int i=0;i<task.count;i++)std::copy_n(task.Goal(m,d,i).begin(),3,d->qpos+m->jnt_qposadr[task.joint[i]]);
  mj_forward(m,d);for(int k=0;k<501;k++)task.Finish(m,d);
  Check(d->userdata[CollectionTask::Outcome]==1,"Placement not recognized");
  task.Reset(m,d,12);d->qpos[2]=task.fallHeight-.1;mj_forward(m,d);task.Finish(m,d);Check(d->userdata[CollectionTask::Outcome]==2,"Fall not recognized");
  mj_resetData(m,d);task.Reset(m,d,12);d->userdata[CollectionTask::Elapsed]=90;task.Finish(m,d);Check(d->userdata[CollectionTask::Outcome]==3,"Timeout not recognized");
  std::cout<<name<<" 128 starts, scoring passed\n";mj_deleteData(d);mj_deleteModel(m);
 }
 char err[2048]{};auto path=std::string(argv[1])+"/scene-tennis.xml";
 auto* m=mj_loadXML(path.c_str(),nullptr,err,sizeof err);if(!m)throw std::runtime_error(err);auto* d=mj_makeData(m);TennisTask t;t.Initialize(m);
 double minNet=10,maxNet=0,minApex=10,maxApex=0,minTime=10,maxTime=0,minY=10,maxY=-10,minHitZ=10,maxHitZ=0;
 for(uint32_t seed=0;seed<64;seed++){
  t.Reset(m,d);t.SetSeed(m,d,seed);auto* s=d->userdata;TennisTask::Vec v{s[TennisTask::LaunchX],s[TennisTask::LaunchY],s[TennisTask::LaunchZ]},p{3,0,1.32};
  double mat[9];mju_quat2Mat(mat,d->mocap_quat+4*t.aimMocap);auto dir=TennisTask::Unit(v);
  for(int k=0;k<3;k++)Check(std::abs(mat[3*k+2]-dir[k])<1e-12,"Barrel does not match shot");
  const double targetX=s[TennisTask::BounceX],targetY=s[TennisTask::BounceY];minY=std::min(minY,targetY);maxY=std::max(maxY,targetY);
  double apex=0,time=0;bool bounced=false,crossed=false;
  for(int k=0;k<20000;k++){
   auto old=p;TennisTask::Fly(p,v,.0001);time+=.0001;apex=std::max(apex,p[2]);
   if(old[0]>TennisTask::Mid&&p[0]<=TennisTask::Mid){crossed=true;minNet=std::min(minNet,p[2]);maxNet=std::max(maxNet,p[2]);Check(p[2]>.95,"Net clearance");}
   if(!bounced&&p[2]<=.78&&v[2]<0){
    Check(crossed,"Bounced before net");Check(std::abs(p[0]-targetX)<.002&&std::abs(p[1]-targetY)<.002,"Wrong bounce target");
    v[2]=-.9*v[2];p[2]=.78;bounced=true;minTime=std::min(minTime,time);maxTime=std::max(maxTime,time);
   }
   if(bounced&&p[0]<.40){minHitZ=std::min(minHitZ,p[2]);maxHitZ=std::max(maxHitZ,p[2]);Check(p[2]>.85&&p[2]<1.6,"Unreachable return height");break;}
  }
  Check(bounced,"No bounce");minApex=std::min(minApex,apex);maxApex=std::max(maxApex,apex);
 }
 // The announced vector is exactly the fired vector. During the flight the
 // next shot is prepared ahead of the deadline and survives an mjData copy.
 t.Reset(m,d);t.SetSeed(m,d,42);d->userdata[TennisTask::Armed]=1;d->userdata[TennisTask::NextServe]=7;
 TennisTask::Vec announced{d->userdata[TennisTask::LaunchX],d->userdata[TennisTask::LaunchY],d->userdata[TennisTask::LaunchZ]};
 d->time=7;t.Begin(d);t.Finish(m,d);
 for(int k=0;k<3;k++)Check(std::abs(t.velocity[k]-announced[k])<1e-12,"Fired a different direction than preview");
 d->time=10.49;t.Begin(d);t.Finish(m,d);Check(d->userdata[TennisTask::PreparedServe]==1,"Changed preview too soon during flight");
 d->time=10.5;t.Begin(d);t.Finish(m,d);Check(d->userdata[TennisTask::PreparedServe]==2,"No advance preview");
 Check(std::abs(d->userdata[TennisTask::LaunchY]-announced[1])>.001,"Repeated identical serve");
 auto* copied=mj_copyData(nullptr,m,d);Check(std::equal(d->userdata,d->userdata+m->nuserdata,copied->userdata),"Shot snapshot lost state");Check(std::equal(d->mocap_quat,d->mocap_quat+4,copied->mocap_quat),"Snapshot lost aim");mj_deleteData(copied);
 Check(minY<-.30&&maxY>.30,"Missing lateral variety");
 std::cout<<"tennis 64 seeds: net="<<minNet<<".."<<maxNet<<" apex="<<minApex<<".."<<maxApex<<" time_to_bounce="<<minTime<<".."<<maxTime<<" return_height="<<minHitZ<<".."<<maxHitZ<<"\n";
 mj_deleteData(d);mj_deleteModel(m);return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

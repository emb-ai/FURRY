// Synthetic pipeline fixture, never a human demonstration or policy benchmark.
#include "recording.h"
#include "collection_task.h"
#include <iostream>
int main(int argc,char** argv){
 try{
  if(argc!=3)return 2;
  std::filesystem::path assets=argv[1],root=argv[2];std::filesystem::create_directories(root);
  char error[2048];auto* m=mj_loadXML((assets/"scene-cup.xml").c_str(),nullptr,error,sizeof(error));
  if(!m)throw std::runtime_error(error);auto*d=mj_makeData(m);CollectionTask task;task.Initialize(m);
  std::filesystem::copy_file(assets/"recording_metadata-cup.json",root/"recording_metadata.json",std::filesystem::copy_options::overwrite_existing);
  for(int trial=0;trial<4;trial++){
   mj_resetData(m,d);task.Reset(m,d,100+trial);EpisodeRecorder rec;rec.Start(root.string(),m,"recording_metadata.json","",false,true);
   std::ofstream session(rec.path()+"/session_config.json");session<<"{\"capture\":\"human_and_robot\",\"scene\":\"cup\",\"fixture\":true}";session.close();
   TrackingFrame f;f.valid=true;f.location_flags={15,15,15};f.hand_active={true,true};
   f.body.supported=f.body.valid=true;f.body.confidence=1;
   std::array<float,29> ref{};std::array<float,35> cmd{};cmd[3]=.75;
   int q=m->jnt_qposadr[task.joint[0]];double sx=d->qpos[q],sy=d->qpos[q+1],sz=d->qpos[q+2];
   const auto target=task.Goal(m,d,0);
   for(int i=0;i<401;i++){
    double t=i*.01;d->time=t;f.sequence=i+1;f.xr_time_ns=1000000000LL+i*10000000LL;
    // Direct kinematic motion only: lift, carry, release, settle at goal.
    double lift=std::min(1.,std::max(0.,t/.5));double carry=std::min(1.,std::max(0.,(t-.5)/1.));double lower=std::min(1.,std::max(0.,(t-1.5)/.5));
    if(trial!=1){d->qpos[q]=sx+(target[0]-sx)*carry;d->qpos[q+1]=sy+(target[1]-sy)*carry;d->qpos[q+2]=sz+.10*lift+lower*(target[2]-sz-.10);}
    mj_forward(m,d);for(int k=0;k<10;k++)task.Finish(m,d);
    double look[3]={1,0,-.45};mju_normalize3(look);rec.LearningGazeWorld(look,true);
    if(trial==2&&i==150)rec.Event("user_pause",f.sequence);
    rec.Input(f,.2,0,EpisodeRecorder::Now());
    rec.Frame(m,d,i*10,f,false,false,!(trial==3&&i==150),true,.2,0,1,0,false,ref);
    cmd[0]=float(.01*std::sin(t));rec.Mimic(f,i*10,t,cmd);
   }
   rec.LearningFinalState(m,d);rec.Stop();std::cout<<rec.path()<<'\n';
  }
  mj_deleteData(d);mj_deleteModel(m);return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}

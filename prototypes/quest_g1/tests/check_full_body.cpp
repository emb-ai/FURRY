#include "full_body_io.h"
#include "recording.h"
#include <iostream>
#include <limits>
void Require(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
template<class F> void Reject(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}Require(bad,"Invalid input accepted");}
int main(int argc,char**argv){
 if(argc!=2)return 2;
 TrackingFrame f;f.sequence=42;f.xr_time_ns=123456789012345;f.body.time_ns=f.xr_time_ns-100;f.body.supported=true;f.body.valid=true;f.body.confidence=.875;
 auto bind=std::make_shared<FullBodySkeleton>();bind->version=7;
 auto full=std::make_shared<FullBodyFrame>();full->version=7;full->active=true;full->flags.fill(15);full->skeleton=bind;
 for(int j=0;j<84;j++){
   full->joints[j].position={double(float(.01*j)),double(float(1.+.003*j)),double(float(-.3))};
   bind->parent[j]=j-1;
 }
 bind->rest=full->joints;f.body.full=full;f.body.joints[0]=full->joints[1];f.body.flags.fill(15);
 fullbody::Recorder writer;writer.Start(argv[1]);writer.Input(f,987654321012345);writer.Flush();
 auto second=f;second.sequence++;second.xr_time_ns++;auto next=std::make_shared<FullBodyFrame>(*full);next->active=false;next->flags[83]=0;next->flags[3]=uint64_t(1)<<60;next->joints[83].position[0]=std::numeric_limits<double>::quiet_NaN();second.body.full=next;writer.Input(second,123);
 auto third=second;third.sequence++;third.xr_time_ns++;third.body.full.reset();writer.Input(third,124);writer.Close();
 Require(std::filesystem::file_size(std::filesystem::path(argv[1])/"body_full.bin")==20+3*3072,"Unexpected frame wire size");
 fullbody::Reader reader(argv[1],true);auto decoded=f;decoded.body.full.reset();reader.Attach(decoded);
 Require(decoded.body.full&&decoded.body.full->skeleton->parent==bind->parent,"Skeleton topology lost");
 for(int j=0;j<84;j++)for(int k=0;k<3;k++)Require(decoded.body.full->joints[j].position[k]==float(full->joints[j].position[k]),"Raw position changed");
 decoded=second;decoded.body.full.reset();reader.Attach(decoded);Require(!decoded.body.full->active&&decoded.body.full->flags[3]==(uint64_t(1)<<60)&&std::isnan(decoded.body.full->joints[83].position[0]),"Inactive/raw flags/NaN lost");
 decoded=third;reader.Attach(decoded);Require(!decoded.body.full,"Missing sample fabricated");Reject([&]{reader.Attach(decoded);});
 fullbody::Reader wrong(argv[1],true);decoded=f;decoded.sequence=43;Reject([&]{wrong.Attach(decoded);});
 fullbody::Reader legacy(std::string(argv[1])+"/absent");decoded=f;decoded.body.full.reset();legacy.Attach(decoded);Require(!decoded.body.full,"Legacy stream fabricated full body");Reject([&]{fullbody::Reader missing(std::string(argv[1])+"/absent",true);});

 // Exercise the actual episode writer with physics paused, as in human capture.
 auto folder=std::filesystem::path(argv[1]);
 std::ofstream(folder/"model.xml") << "<mujoco><worldbody><body pos='0 0 1'><freejoint/><geom type='sphere' size='.1'/></body></worldbody></mujoco>";
 std::ofstream(folder/"recording_metadata.json") << "{}";
 char error[1024]{};auto*m=mj_loadXML((folder/"model.xml").c_str(),nullptr,error,sizeof(error));Require(m,error);auto*d=mj_makeData(m);
 EpisodeRecorder episode;episode.Start(argv[1],m);episode.Input(f,0,0,123);
 episode.Frame(m,d,0,f,false,true,false,true,0,0,0,0,false,{});episode.Stop();
 auto saved=std::filesystem::path(episode.path());
 Require(std::filesystem::file_size(saved/"body_full.bin")==20+3072,"Episode did not include full body");
 Require(std::filesystem::exists(saved/"complete.json"),"Successful episode not finalized");
 fullbody::Reader complete(saved.string(),true);decoded=f;decoded.body.full.reset();complete.Attach(decoded);Require(decoded.body.full&&decoded.body.full->skeleton,"Episode skeleton missing");
 episode.Start(argv[1],m);episode.Input(f,0,0,123);auto interrupted=std::filesystem::path(episode.path());episode.Abort();
 Require(!std::filesystem::exists(interrupted/"complete.json"),"Aborted episode marked complete");
 mj_deleteData(d);mj_deleteModel(m);
 std::cout<<"Raw full-body roundtrip, exact flags/timestamps, missing frames and real episode lifecycle passed\n";
}

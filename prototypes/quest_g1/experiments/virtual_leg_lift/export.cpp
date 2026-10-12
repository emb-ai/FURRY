#include "meta_retarget.h"
#include "simulation.h"
#include "episode_manifest.h"
#include <fstream>
#include <sstream>
#include <map>
#include <iostream>
#include <iomanip>
static std::vector<double> Row(const std::string&s){std::stringstream in(s);std::string cell;std::vector<double> out;while(std::getline(in,cell,','))out.push_back(std::stod(cell));return out;}
int main(int argc,char**argv){try{
 if(argc!=5)return 2;std::string assets=argv[1],folder=argv[2],out=argv[3];bool patch=std::string(argv[4])=="on";
 auto manifest=episodereplay::LoadManifest(folder);Simulation sim(assets,0,"stand",assets+"/policy-quest-111586-500.onnx",50);
 MetaRetargeter meta(assets);meta.EnableNeutralWrists(true);meta.EnableWristFrameV2(true);meta.EnableCameraTracking(manifest.firstPerson);meta.SetTravelGain(manifest.travelGainPercent/100.);meta.EnableVirtualLegLift(patch);
 std::map<uint64_t,TrackingFrame> poses;std::string line;
 std::ifstream input(folder+"/input.csv"),body(folder+"/body.csv"),frames(folder+"/frames.csv");
 std::getline(input,line);while(std::getline(input,line)){auto r=Row(line);TrackingFrame f;f.sequence=r[0];f.xr_time_ns=r[1];f.valid=r[3];for(int j=0;j<3;j++)f.location_flags[j]=r[4+j];f.hand_active={r[7]!=0,r[8]!=0};TrackedPose*p[]={&f.head,&f.hands[0],&f.hands[1]};for(int j=0;j<3;j++){std::copy_n(r.begin()+11+7*j,3,p[j]->position.begin());std::copy_n(r.begin()+14+7*j,4,p[j]->quaternion.begin());}poses[f.sequence]=f;}
 std::getline(body,line);while(std::getline(body,line)){auto r=Row(line);auto&b=poses.at(r[0]).body;b.time_ns=r[1];b.supported=r[2];b.valid=r[3];b.confidence=r[4];b.skeleton_version=r[5];for(int j=0;j<14;j++){int a=6+15*j;b.flags[j]=r[a];for(int k=0;k<2;k++){auto&p=k?b.rest[j]:b.joints[j];std::copy_n(r.begin()+a+1+7*k,3,p.position.begin());std::copy_n(r.begin()+a+4+7*k,4,p.quaternion.begin());}}}
 std::ofstream csv(out);csv<<std::setprecision(17)<<"time,sequence,swing,left_lift,right_lift,left_raw_clearance,right_raw_clearance,ik_limited";
 for(int i=0;i<35;i++)csv<<",cmd_"<<i;for(int i=0;i<36;i++)csv<<",ref_"<<i;for(int i=0;i<7;i++)csv<<",goal_"<<i;
 for(auto prefix:{"raw","patched"})for(int j=0;j<14;j++)for(int k=0;k<7;k++)csv<<','<<prefix<<'_'<<j<<'_'<<k;
 csv<<'\n';double start=-1;int n=0;std::getline(frames,line);
 while(std::getline(frames,line)){
  auto r=Row(line);auto&f=poses.at(r[1]);
  if(r[5]){std::copy_n(r.begin()+42,sim.model->nq,sim.data->qpos);mj_forward(sim.model,sim.data);meta.Calibrate(sim.model,sim.data,f);continue;}
  if(!r[6]){meta.Pause();continue;}if(!meta.calibrated)throw std::runtime_error("Missing calibration");if(start<0)start=r[3];
  auto cmd=meta.Solve(f);const auto&d=meta.LiftDiagnostics();auto goal=meta.solver().HasCameraTarget()?meta.solver().CameraTarget():meta.solver().CameraPose();
  csv<<r[3]-start<<','<<f.sequence<<','<<d.swingSide<<','<<d.lift[0]<<','<<d.lift[1]<<','<<d.rawClearance[0]<<','<<d.rawClearance[1]<<','<<d.reachLimited;
  for(auto v:cmd)csv<<','<<v;for(int k=0;k<36;k++)csv<<','<<meta.solver().data()->qpos[k];for(auto v:goal.position)csv<<','<<v;for(auto v:goal.quaternion)csv<<','<<v;
  for(auto b:{f.body,meta.CorrectedBody()})for(auto p:b.joints){for(auto v:p.position)csv<<','<<v;for(auto v:p.quaternion)csv<<','<<v;}
  csv<<'\n';n++;
 }
 std::cout<<"patch="<<patch<<" samples="<<n<<'\n';
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

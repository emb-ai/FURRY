// Offline camera-leg replay: recorded Meta body + recorded camera packets ->
// the live CameraFusion -> native GMR, on the recorded physics ticks.
// Writes the actor commands in mimic.csv format (comparable to the recorded
// mimic.csv when the episode drove camera legs) and, per applying tick, the
// GMR robot pose and the fusion state. <out>.fused.txt holds the fused
// body of the first tick of every input frame in retarget_capture's format.
// packets "-" replays Meta legs. An optional body file in the fused.txt
// format replaces the recorded body of its frames and bypasses fusion (the
// cleaned offline target goes through the same calibration and GMR).
// Usage: camera_commands assets episode packets.txt|- registration.txt|- out_prefix [body.txt]
#include "meta_retarget.h"
#include "simulation.h"
#include "episode_manifest.h"
#include "camera_fusion.h"
#include "camera_pose_servo.h"
#include <fstream>
#include <sstream>
#include <map>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
static std::vector<double> Row(const std::string& line){std::stringstream s(line);std::string item;std::vector<double> row;while(std::getline(s,item,',')){double v=std::stod(item);if(!std::isfinite(v))throw std::runtime_error("Nonfinite recording");row.push_back(v);}return row;}
struct Packet{double observedNs,observedEpochMs;CameraSkeleton skeleton;};
static std::vector<Packet> Packets(const std::string& path){
    std::ifstream in(path);if(!in)throw std::runtime_error("Cannot open packets");std::vector<Packet> out;std::string line;
    auto joint=[](std::istream&s,CameraJoint&j){int measured;s>>j.p[0]>>j.p[1]>>j.p[2]>>j.confidence>>measured;j.measured=measured!=0;};
    while(std::getline(in,line)){
        std::stringstream s(line);Packet p;int camera;
        s>>p.observedNs>>p.observedEpochMs>>p.skeleton.receivedMs>>p.skeleton.sourceMs>>p.skeleton.sequence>>camera;
        p.skeleton.frame=camera?"camera":"pelvis-relative";joint(s,p.skeleton.pelvis);for(auto&j:p.skeleton.joints)joint(s,j);
        if(!s)throw std::runtime_error("Invalid packet row");out.push_back(p);
    }
    return out;
}
int main(int argc,char**argv){
    if(argc!=6 && argc!=7){std::cerr<<"Usage: camera_commands assets episode packets.txt|- registration.txt|- out_prefix [body.txt]\n";return 2;}
    try{
        std::string folder=argv[2],prefix=argv[5];auto manifest=episodereplay::LoadManifest(folder);
        Simulation sim(argv[1],2,manifest.scene,manifest.Policy(argv[1],folder),manifest.policyHz);manifest.ValidateDimensions(sim.model->nq,sim.model->nv,sim.model->nu);MetaRetargeter meta(argv[1]);meta.EnableVirtualLegLift(manifest.inputPatch=="virtual_leg_lift10_v1");meta.EnableNeutralWrists(manifest.neutralWrists);meta.EnableWristFrameV2(manifest.wristFrameV2);meta.EnableControllerWrists(manifest.scene=="rubik");meta.SetTravelGain(manifest.travelGainPercent/100.);
        std::vector<Packet> packets;double epochOffset=0;
        if(std::string(argv[3])!="-"){
            packets=Packets(argv[3]);if(packets.empty())throw std::runtime_error("No camera packets");
            // Ticks carry steady-clock time; fusion runs on the system clock.
            std::vector<double> offsets;for(auto&p:packets)offsets.push_back(p.observedEpochMs-p.observedNs*1e-6);
            std::nth_element(offsets.begin(),offsets.begin()+offsets.size()/2,offsets.end());epochOffset=offsets[offsets.size()/2];
        }
        CameraFusion fusion;
        if(std::string(argv[4])!="-" && !fusion.LoadRegistration(argv[4]))throw std::runtime_error("Invalid camera registration");
        std::ifstream inputs(folder+"/input.csv"),bodies(folder+"/body.csv"),frames(folder+"/frames.csv"),commands(folder+"/mimic.csv");
        if(!inputs||!bodies||!frames||!commands)throw std::runtime_error("Replay requires input/body/frames/mimic streams");
        std::map<uint64_t,TrackingFrame> poses;std::string line;
        std::getline(inputs,line);while(std::getline(inputs,line)){
            auto r=Row(line);if(r.size()!=32)throw std::runtime_error("Invalid input row");
            TrackingFrame f;f.sequence=r[0];f.xr_time_ns=r[1];f.valid=r[3];for(int j=0;j<3;j++)f.location_flags[j]=r[4+j];f.hand_active={r[7]!=0,r[8]!=0};
            TrackedPose* p[]={&f.head,&f.hands[0],&f.hands[1]};for(int i=0;i<3;i++){std::copy_n(r.begin()+11+7*i,3,p[i]->position.begin());std::copy_n(r.begin()+14+7*i,4,p[i]->quaternion.begin());}poses[f.sequence]=f;
        }
        std::getline(bodies,line);while(std::getline(bodies,line)){
            auto r=Row(line);if(r.size()!=216 || !poses.count(r[0]))throw std::runtime_error("Invalid body row");auto&b=poses.at(r[0]).body;
            b.time_ns=r[1];b.supported=r[2];b.valid=r[3];b.confidence=r[4];b.skeleton_version=r[5];
            for(int i=0;i<14;i++){int a=6+15*i;b.flags[i]=r[a];for(int k=0;k<2;k++){auto&p=k?b.rest[i]:b.joints[i];std::copy_n(r.begin()+a+1+7*k,3,p.position.begin());std::copy_n(r.begin()+a+4+7*k,4,p.quaternion.begin());}}
        }
        bool replaced=argc==7;
        if(replaced){
            std::ifstream in(argv[6]);if(!in)throw std::runtime_error("Cannot open body override");int rows=0;
            while(std::getline(in,line)){
                std::stringstream row(line);uint64_t sequence;double skip;row>>sequence;
                auto it=poses.find(sequence);if(it==poses.end())throw std::runtime_error("Body override for unknown frame");
                for(int k=0;k<10;k++)row>>skip;
                for(int j=0;j<14;j++){auto&b=it->second.body;for(auto*p:{&b.joints[j],&b.rest[j]}){for(double&v:p->position)row>>v;for(double&v:p->quaternion)row>>v;}}
                if(!row)throw std::runtime_error("Invalid body override row");rows++;
            }
            std::cerr<<"body override rows="<<rows<<'\n';
        }
        auto focusPauses=episodereplay::PauseTimes(folder);
        auto viewChanges=episodereplay::ViewChanges(folder);size_t viewIndex=0,pauseIndex=0,next=0;bool cameraTracking=manifest.firstPerson;
        int egoCamera=mj_name2id(sim.model,mjOBJ_CAMERA,"ego");
        std::array<float,35> origin{};double blendStart=0;bool wasApplying=false;int count=0,cameraTicks=0;
        std::ofstream mimic(prefix+".mimic.csv"),pose(prefix+".pose.csv"),body(prefix+".fused.txt"),fusedState(prefix+".fused.csv");uint64_t lastFused=~0ull;
        fusedState<<"sequence,receive_ns,camera_legs,camera_weight,camera_state,apply_reference\n";body<<std::setprecision(17);
        std::getline(frames,line);std::string command;std::getline(commands,command);
        mimic<<command<<'\n';pose<<"sequence,step,sim_time,receive_ns,camera_legs,camera_weight,camera_state";for(int k=0;k<36;k++)pose<<",q"<<k;pose<<'\n';
        auto emit=[&](const std::vector<double>& c,const float* v){mimic<<uint64_t(c[0])<<","<<uint64_t(c[1])<<","<<std::setprecision(17)<<c[2]<<std::setprecision(9);for(int i=0;i<35;i++)mimic<<","<<(v?double(v[i]):c[3+i]);mimic<<"\n";};
        while(std::getline(frames,line)){
            if(!std::getline(commands,command))throw std::runtime_error("Missing mimic frame");auto r=Row(line),c=Row(command);
            if(r.size()!=size_t(42+sim.model->nq+sim.model->nv+sim.model->nu)||c.size()!=38 || c[0]!=r[1]||c[1]!=r[2]||c[2]!=r[3])throw std::runtime_error("Mismatched frame/mimic streams");
            while(pauseIndex<focusPauses.size() && focusPauses[pauseIndex]<=r[0]){wasApplying=false;meta.Pause();pauseIndex++;}
            while(viewIndex<viewChanges.size()&&viewChanges[viewIndex].first<=r[0])cameraTracking=viewChanges[viewIndex++].second;
            meta.EnableCameraTracking(cameraTracking);
            auto it=poses.find(r[1]);if(it==poses.end())throw std::runtime_error("Missing raw frame");const auto&f=it->second;
            std::copy_n(r.begin()+42,sim.model->nq,sim.data->qpos);mj_forward(sim.model,sim.data);
            // The packet a tick saw is recorded in that tick, before fusion.
            while(next<packets.size() && packets[next].observedNs<=r[0])next++;
            const CameraSkeleton* latest=next?&packets[next-1].skeleton:nullptr;
            if(r[4]){meta.calibrated=false;wasApplying=false;fusion.Reset();}
            if(r[5]){meta.Calibrate(sim.model,sim.data,f);fusion.Reset();}
            TrackingFrame fused=f;
            bool valid=f.body.valid && (f.location_flags[0]&3)==3 && std::abs(double(f.xr_time_ns)-double(f.body.time_ns))<2e8;
            if(valid && !replaced){double now=r[0]*1e-6+epochOffset;fusion.Observe(f,now);fused=fusion.Apply(f,latest,now);}
            if(valid && f.sequence!=lastFused){
                lastFused=f.sequence;const auto&b=fused.body;
                body<<f.sequence<<' '<<f.xr_time_ns<<' '<<b.time_ns<<' '<<b.skeleton_version;
                for(double v:f.head.position)body<<' '<<v;for(double v:f.head.quaternion)body<<' '<<v;
                for(int j=0;j<14;j++)for(auto*p:{&b.joints[j],&b.rest[j]}){for(double v:p->position)body<<' '<<v;for(double v:p->quaternion)body<<' '<<v;}
                body<<'\n';fusedState<<f.sequence<<','<<uint64_t(r[0])<<','<<fusion.stats.legs<<','<<std::setprecision(9)<<fusion.stats.weight<<','<<fusion.stats.state<<','<<int(r[6])<<'\n';
            }
            if(!r[6]){wasApplying=false;meta.Pause();emit(c,nullptr);continue;}
            if(!meta.calibrated){emit(c,nullptr);continue;}
            if(r[5]||!wasApplying){origin.fill(0);origin[2]=sim.data->qpos[2];blendStart=r[3];auto*gm=meta.solver().model();for(int j=1;j<gm->njnt;j++){int source=mj_name2id(sim.model,mjOBJ_JOINT,mj_id2name(gm,mjOBJ_JOINT,j));origin[gm->jnt_qposadr[j]-1]=sim.data->qpos[sim.model->jnt_qposadr[source]];}}
            auto result=meta.Solve(fused);
            auto* d=meta.solver().data();
            pose<<uint64_t(r[1])<<','<<uint64_t(r[2])<<','<<std::setprecision(17)<<r[3]<<','<<uint64_t(r[0])<<','<<fusion.stats.legs<<','<<fusion.stats.weight<<','<<fusion.stats.state;
            for(int k=0;k<36;k++)pose<<','<<d->qpos[k];pose<<'\n';
            if(manifest.catchUp){auto& solver=meta.solver();result=ApplyCameraCatchUp(result,solver.HasCameraTarget()?solver.CameraTarget():solver.CameraPose(),sim.model,sim.data,egoCamera,true,manifest.catchUpV2?solver.data()->qpos+3:nullptr);}
            double u=std::clamp((r[3]-blendStart)/.5,0.,1.);
            for(int i=0;i<35;i++)result[i]=origin[i]+u*(result[i]-origin[i]);
            wasApplying=true;count++;cameraTicks+=fusion.stats.legs>0;emit(c,result.data());
        }
        if(std::getline(commands,command))throw std::runtime_error("Extra mimic rows");
        std::cout<<"camera replay frames="<<count<<" camera_leg_ticks="<<cameraTicks<<" packets="<<packets.size()<<'\n';return count>0?0:1;
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}
}

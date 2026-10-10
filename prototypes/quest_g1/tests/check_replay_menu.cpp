#include "meta_retarget.h"
#include "simulation.h"
#include "recording.h"
#include "policy_catalog.h"
#include "camera_pose_servo.h"
#include <iostream>

// Synthetic source poses only. Exercise per-scene streams, menu/user/focus
// pauses and calibration snapshots without any participant trajectory.
int main(int argc,char** argv){
    if(argc<4||argc>7)return 2;
    bool catchUp=argc>=6&&std::string(argv[5])=="on",firstPerson=argc>=7&&std::string(argv[6])=="first";
    bool switchView=argc>=7&&std::string(argv[6])=="switch";
    try{
        std::string name=argv[3],policyPath,metadata=name=="lab"?"recording_metadata.json":"recording_metadata-"+name+".json",policyFile;
        if(argc>=5){
            auto catalog=questpolicy::Load(argv[1]);bool found=false;
            for(const auto& entry:catalog)if(entry.id==argv[4]){policyPath=questpolicy::VerifiedPath(argv[1],entry);metadata=questpolicy::Metadata(name,entry);policyFile=entry.file;found=true;}
            if(!found)throw std::runtime_error("Unknown test policy");
        }
        Simulation sim(argv[1],2,name,policyPath);MetaRetargeter meta(argv[1]);meta.EnableNeutralWrists(true);auto& solver=meta.solver();
        meta.EnableCameraTracking(firstPerson);int egoCamera=mj_name2id(sim.model,mjOBJ_CAMERA,"ego");
        auto* m=solver.model();std::unique_ptr<mjData,decltype(&mj_deleteData)> data(mj_makeData(m),mj_deleteData);auto* d=data.get();
        TrackingFrame input;input.valid=input.body.valid=input.body.supported=true;input.location_flags={15,15,15};input.hand_active={true,true};input.body.confidence=1;input.head.position={0,1.65,0};
        auto fill=[&](bool rest){
            auto& poses=rest?input.body.rest:input.body.joints;
            for(int i=0;i<14;i++){
                int body=solver.tasks()[i].body;
                if(i==6||i==7)body=mj_name2id(m,mjOBJ_BODY,i==6?"left_ankle_roll_link":"right_ankle_roll_link");
                if(i==8||i==9)body=mj_name2id(m,mjOBJ_BODY,i==8?"left_shoulder_pitch_link":"right_shoulder_pitch_link");
                const auto* p=d->xpos+3*body;poses[i].position={-p[1]*1.25,p[2]*1.25,-p[0]*1.25};
                double basis[9]={0,-1,0,0,0,1,-1,0,0},rotation[9];mju_mulMatMat(rotation,basis,d->xmat+9*solver.tasks()[i].body,3,3,3);mju_mat2Quat(poses[i].quaternion.data(),rotation);input.body.flags[i]=15;
            }
        };
        mj_resetData(m,d);d->qpos[2]=.8;
        for(auto side:{"left","right"}){
            int roll=mj_name2id(m,mjOBJ_JOINT,(std::string(side)+"_shoulder_roll_joint").c_str()),elbow=mj_name2id(m,mjOBJ_JOINT,(std::string(side)+"_elbow_joint").c_str());
            d->qpos[m->jnt_qposadr[roll]]=(std::string(side)=="left"?1:-1)*1.57079632679;d->qpos[m->jnt_qposadr[elbow]]=1.57079632679;
        }
        mj_kinematics(m,d);mj_comPos(m,d);fill(true);
        for(int j=1;j<m->njnt;j++){int actual=mj_name2id(sim.model,mjOBJ_JOINT,mj_id2name(m,mjOBJ_JOINT,j));d->qpos[m->jnt_qposadr[j]]=sim.data->qpos[sim.model->jnt_qposadr[actual]];}
        mj_kinematics(m,d);mj_comPos(m,d);fill(false);
        EpisodeRecorder recorder;recorder.Start(argv[2],sim.model,metadata,policyFile,catchUp,firstPerson);
        uint64_t sequence=0;
        auto sample=[&]{input.sequence=++sequence;input.xr_time_ns=input.body.time_ns=1000000000LL+sequence*10000000LL;recorder.Input(input,0,0,EpisodeRecorder::Now());};
        std::array<float,35> origin{};double blendStart=0;
        auto blend=[&]{origin.fill(0);origin[2]=sim.data->qpos[2];blendStart=sim.data->time;for(int j=1;j<m->njnt;j++){int actual=mj_name2id(sim.model,mjOBJ_JOINT,mj_id2name(m,mjOBJ_JOINT,j));origin[m->jnt_qposadr[j]-1]=sim.data->qpos[sim.model->jnt_qposadr[actual]];}};
        auto snapshot=[&](bool reset){
            sample();meta.Calibrate(sim.model,sim.data,input);blend();recorder.Event("calibrate",input.sequence);
            std::array<float,29> joints{};std::copy_n(sim.WholeBodyReference().begin()+6,29,joints.begin());
            recorder.Frame(sim.model,sim.data,sim.steps,input,reset,true,false,true,0,0,0,meta.error,false,joints);
            recorder.Mimic(input,sim.steps,sim.data->time,sim.WholeBodyReference());
        };
        snapshot(true);
        for(int frame=0;frame<160;frame++){
            int shoulder=mj_name2id(m,mjOBJ_JOINT,"left_shoulder_pitch_joint");d->qpos[m->jnt_qposadr[shoulder]]=-.02*std::sin(frame*.01);mj_kinematics(m,d);mj_comPos(m,d);fill(false);
            if(frame==100)snapshot(false);
            sample();
            if(switchView&&(frame==60||frame==110)){firstPerson=frame==60;meta.EnableCameraTracking(firstPerson);recorder.Event(firstPerson?"ego_world":"observer_view",input.sequence);}
            if(frame==40||frame==80||frame==120){
                const char* pause=frame==40?"menu_open":frame==80?"user_pause":"focus_pause";
                recorder.Event(pause,input.sequence);meta.Pause();sim.PauseWholeBodyReference();blend();
                recorder.Event(frame==40?"menu_close":frame==80?"session_resume":"focus_resume",input.sequence);
            }
            auto reference=meta.Solve(input);
            if(catchUp)reference=ApplyCameraCatchUp(reference,solver.HasCameraTarget()?solver.CameraTarget():solver.CameraPose(),sim.model,sim.data,egoCamera,true,solver.data()->qpos+3);
            double amount=std::clamp((sim.data->time-blendStart)/.5,0.,1.);
            for(int k=0;k<35;k++)reference[k]=origin[k]+amount*(reference[k]-origin[k]);sim.SetWholeBodyReference(reference);
            std::array<float,29> joints{};std::copy_n(reference.begin()+6,29,joints.begin());
            recorder.Frame(sim.model,sim.data,sim.steps,input,false,false,true,true,0,0,1,meta.error,false,joints);recorder.Mimic(input,sim.steps,sim.data->time,reference);
            for(int k=0;k<10;k++)sim.Step(false);
        }
        recorder.Stop();std::cout<<recorder.path()<<'\n';return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}

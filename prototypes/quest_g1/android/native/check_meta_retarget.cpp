#include "meta_retarget.h"
#include "simulation.h"
#include "recording.h"
#include <cstdio>
#include <chrono>
// Synthetic fully-observed skeleton, including a bind T pose; this is not
// a claim of validation against actual Meta tracking on a human operator.
int main(int argc,char**argv){
    if(argc<2 || argc>3)return 2;
    Simulation sim(argv[1]);MetaRetargeter meta(argv[1]);meta.EnableNeutralWrists(true);auto& oracle=meta.solver();
    auto*m=oracle.model();auto*d=mj_makeData(m);
    auto fill=[&](TrackingFrame& f,bool rest){
        auto& poses=rest?f.body.rest:f.body.joints;
        for(int i=0;i<14;i++){
            int b=oracle.tasks()[i].body;
            if(i==6||i==7)b=mj_name2id(m,mjOBJ_BODY,i==6?"left_ankle_roll_link":"right_ankle_roll_link");
            if(i==8||i==9)b=mj_name2id(m,mjOBJ_BODY,i==8?"left_shoulder_pitch_link":"right_shoulder_pitch_link");
            // Robot +X,+Y,+Z -> XR -Z,-X,+Y, with a taller source skeleton.
            auto*p=d->xpos+3*b;poses[i].position={-p[1]*1.25,p[2]*1.25,-p[0]*1.25};
            double basis[9]={0,-1,0,0,0,1,-1,0,0},r[9];mju_mulMatMat(r,basis,d->xmat+9*oracle.tasks()[i].body,3,3,3);mju_mat2Quat(poses[i].quaternion.data(),r);f.body.flags[i]=15;
        }
    };
    TrackingFrame input;input.valid=input.body.valid=input.body.supported=true;input.location_flags={15,15,15};input.hand_active={true,true};input.body.confidence=1;input.head.position={0,1.65,0};
    mj_resetData(m,d);d->qpos[2]=.8;
    for(auto side:{"left","right"}){int roll=mj_name2id(m,mjOBJ_JOINT,(std::string(side)+"_shoulder_roll_joint").c_str());int elbow=mj_name2id(m,mjOBJ_JOINT,(std::string(side)+"_elbow_joint").c_str());d->qpos[m->jnt_qposadr[roll]]=(std::string(side)=="left"?1:-1)*1.57079632679;d->qpos[m->jnt_qposadr[elbow]]=1.57079632679;}
    mj_kinematics(m,d);mj_comPos(m,d);fill(input,true);
    for(int j=1;j<m->njnt;j++){int sj=mj_name2id(sim.model,mjOBJ_JOINT,mj_id2name(m,mjOBJ_JOINT,j));d->qpos[m->jnt_qposadr[j]]=sim.data->qpos[sim.model->jnt_qposadr[sj]];}
    mj_kinematics(m,d);mj_comPos(m,d);fill(input,false);meta.Calibrate(sim.model,sim.data,input);
    {
        MetaRetargeter neutral(argv[1]);neutral.EnableNeutralWrists(true);neutral.Calibrate(sim.model,sim.data,input);
        auto sample=input;sample.body.time_ns=1000000000LL;auto command=neutral.Solve(sample);
        for(int k:{25,26,27,32,33,34})if(std::abs(command[k])>1e-5)throw std::runtime_error("Calibrated wrist is not neutral");
        double delta[4]={std::cos(.1),std::sin(.1),0,0},changed[4];mju_mulQuat(changed,sample.body.joints[12].quaternion.data(),delta);
        std::copy_n(changed,4,sample.body.joints[12].quaternion.begin());sample.body.time_ns+=10000000;command=neutral.Solve(sample);
        double angle=std::sqrt(command[25]*command[25]+command[26]*command[26]+command[27]*command[27]);
        if(angle<.18||angle>.22)throw std::runtime_error("Intentional wrist motion was suppressed");
        for(int k:{32,33,34})if(std::abs(command[k])>1e-5)throw std::runtime_error("Opposite wrist moved");
        MetaRetargeter original(argv[1]),reduced(argv[1]);original.EnableCameraTracking(true);reduced.EnableCameraTracking(true);reduced.SetTravelGain(.9);
        original.Calibrate(sim.model,sim.data,input);reduced.Calibrate(sim.model,sim.data,input);sample=input;sample.body.time_ns=1000000000LL;sample.head.position[0]+=.1;
        original.Solve(sample);reduced.Solve(sample);
        auto a=original.solver().CameraTarget(),b=reduced.solver().CameraTarget(),origin=original.CameraCalibrationPose();
        for(int k=0;k<2;k++)if(std::abs((b.position[k]-origin.position[k])-.9*(a.position[k]-origin.position[k]))>1e-9)throw std::runtime_error("Travel gain does not scale target displacement");
        if(std::abs(a.position[2]-b.position[2])>1e-9)throw std::runtime_error("Travel gain changed vertical calibration");
    }
    EpisodeRecorder recorder;
    if(argc==3)recorder.Start(argv[2],sim.model);
    std::array<float,35> blend{};blend[2]=sim.data->qpos[2];
    for(int j=1;j<m->njnt;j++){int sj=mj_name2id(sim.model,mjOBJ_JOINT,mj_id2name(m,mjOBJ_JOINT,j));blend[m->jnt_qposadr[j]-1]=sim.data->qpos[sim.model->jnt_qposadr[sj]];}
    // Independent source-coordinate tests: person size, room transform and
    // device-local bone axes must not change the robot command.
    double maxCoordinateError=0;
    for(int test=0;test<3;test++){
        double scale=test==0?.75:(test==1?1.3:1.),yaw=test==0?.7:(test==1?-1.2:0);
        auto transform=[&](TrackingFrame f){
            double q[4]={std::cos(yaw/2),0,std::sin(yaw/2),0},R[9];mju_quat2Mat(R,q);
            auto pose=[&](TrackedPose& p,bool bone){
                double xyz[3],quat[4];mju_mulMatVec(xyz,R,p.position.data(),3,3);mju_mulQuat(quat,q,p.quaternion.data());
                for(int k=0;k<3;k++)p.position[k]=scale*xyz[k]+(k==0?1.2:(k==2?-2.3:.15));
                if(bone && test==2){double axis[4]={std::cos(.3),std::sin(.3),0,0},updated[4];mju_mulQuat(updated,quat,axis);std::copy_n(updated,4,quat);}
                std::copy_n(quat,4,p.quaternion.begin());
            };
            pose(f.head,false);for(auto& p:f.body.joints)pose(p,true);for(auto& p:f.body.rest)pose(p,true);return f;
        };
        MetaRetargeter a(argv[1]),b(argv[1]);a.Calibrate(sim.model,sim.data,input);b.Calibrate(sim.model,sim.data,transform(input));
        for(int n=0;n<100;n++){
            auto f=input;f.body.time_ns=1000000000LL+n*10000000LL;
            double dx=.04*n/99.;f.head.position[0]+=dx;
            for(auto& p:f.body.joints)p.position[0]+=dx;
            auto ra=a.Solve(f),rb=b.Solve(transform(f));
            for(int k=0;k<35;k++)maxCoordinateError=std::max(maxCoordinateError,std::abs(double(ra[k]-rb[k])));
        }
        auto changed=input;changed.body.skeleton_version++;
        if(!a.Compatible(changed)){std::puts("FAIL unchanged bind skeleton invalidation");return 1;}
        changed.body.rest[6].position[1]+=.02;
        if(a.Compatible(changed)){std::puts("FAIL changed bind skeleton accepted");return 1;}
        a.Pause();auto resumed=input;resumed.body.time_ns=5000000000LL;auto cmd=a.Solve(resumed);
        if(cmd[0]!=0 || cmd[1]!=0 || cmd[5]!=0){std::puts("FAIL resume reference velocity");return 1;}
    }
    std::printf("coordinate/scale/local-axis equivalence max=%.9g\n",maxCoordinateError);
    if(maxCoordinateError>1e-5)return 1;
    auto start=std::chrono::steady_clock::now();double minHeight=1,maxVelocity=0,blendStart=0;
    for(int frame=0;frame<1000;frame++){
        input.sequence=frame+1;input.body.time_ns=input.xr_time_ns=1000000000LL+frame*10000000LL;
        int shoulder=mj_name2id(m,mjOBJ_JOINT,"left_shoulder_pitch_joint");
        d->qpos[m->jnt_qposadr[shoulder]]=-.2*std::sin(frame*.01);
        mj_kinematics(m,d);mj_comPos(m,d);fill(input,false);
        recorder.Input(input,0,0,EpisodeRecorder::Now());
        if(frame==0)recorder.Event("calibrate",input.sequence);
        if(frame==500){
            recorder.Event("focus_pause",input.sequence);meta.Pause();
            recorder.Event("focus_resume",input.sequence);blendStart=sim.data->time;
            blend.fill(0);blend[2]=sim.data->qpos[2];
            for(int j=1;j<m->njnt;j++){int sj=mj_name2id(sim.model,mjOBJ_JOINT,mj_id2name(m,mjOBJ_JOINT,j));blend[m->jnt_qposadr[j]-1]=sim.data->qpos[sim.model->jnt_qposadr[sj]];}
        }
        auto reference=meta.Solve(input);
        if(frame==0)std::printf("initial GMR z=%.4f hip=%.4f knee=%.4f shoulder=%.4f elbow=%.4f\n",reference[2],reference[6],reference[9],reference[21],reference[24]);
        maxVelocity=std::max(maxVelocity,double(std::hypot(reference[0],reference[1])));
        double u=std::min(1.,(sim.data->time-blendStart)/.5);for(int k=0;k<35;k++)reference[k]=blend[k]+u*(reference[k]-blend[k]);
        sim.SetWholeBodyReference(reference);
        std::array<float,29> joints;std::copy_n(reference.begin()+6,29,joints.begin());
        recorder.Frame(sim.model,sim.data,sim.steps,input,false,frame==0,true,true,0,0,1,meta.error,false,joints);
        recorder.Mimic(input,sim.steps,sim.data->time,reference);
        try{for(int k=0;k<10;k++)sim.Step(false);}catch(const std::exception&e){std::printf("%s\n",e.what());return 1;}
        minHeight=std::min(minHeight,sim.data->qpos[2]);
    }
    double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::printf("min_height=%.4f drift=%.4f max_reference_velocity=%.4f seconds=%.3f\n",minHeight,std::hypot(sim.data->qpos[0],sim.data->qpos[1]),maxVelocity,elapsed);
    recorder.Stop();if(argc==3)std::printf("episode=%s\n",recorder.path().c_str());
    mj_deleteData(d);return minHeight>.7 && maxVelocity<1 ? 0:1;
}

#include "meta_retarget.h"
#include "simulation.h"
#include <iostream>
static void Check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static double PositionError(const TrackedPose& a,const TrackedPose& b){double v[3];mju_sub3(v,a.position.data(),b.position.data());return mju_norm3(v);}
static double RotationError(const double* a,const double* b){double inv[4],q[4],v[3];mju_negQuat(inv,a);mju_mulQuat(q,inv,b);mju_quat2Vel(v,q,1);return mju_norm3(v);}
int main(int argc,char** argv){
    try{
        Check(argc==2,"Expected exported assets");Simulation sim(argv[1],0,"rubik");MetaRetargeter meta(argv[1]);
        meta.EnableNeutralWrists(true);meta.EnableWristFrameV2(true);meta.EnableControllerWrists(true);
        auto& solver=meta.solver();auto* m=solver.model();auto* d=mj_makeData(m);
        TrackingFrame input;input.valid=input.body.valid=input.body.supported=true;input.location_flags={15,15,15};
        input.hand_active={true,true};input.body.confidence=1;input.head.position={0,1.65,0};
        auto fill=[&](bool rest){
            auto& poses=rest?input.body.rest:input.body.joints;
            for(int i=0;i<14;i++){
                int body=solver.tasks()[i].body;
                if(i==6||i==7)body=mj_name2id(m,mjOBJ_BODY,i==6?"left_ankle_roll_link":"right_ankle_roll_link");
                if(i==8||i==9)body=mj_name2id(m,mjOBJ_BODY,i==8?"left_shoulder_pitch_link":"right_shoulder_pitch_link");
                const auto* p=d->xpos+3*body;poses[i].position={-p[1]*1.25,p[2]*1.25,-p[0]*1.25};
                double basis[9]={0,-1,0,0,0,1,-1,0,0},rotation[9];mju_mulMatMat(rotation,basis,d->xmat+9*solver.tasks()[i].body,3,3,3);
                mju_mat2Quat(poses[i].quaternion.data(),rotation);input.body.flags[i]=15;
            }
        };
        mj_resetData(m,d);d->qpos[2]=.8;
        for(auto side:{"left","right"}){
            int roll=mj_name2id(m,mjOBJ_JOINT,(std::string(side)+"_shoulder_roll_joint").c_str());
            int elbow=mj_name2id(m,mjOBJ_JOINT,(std::string(side)+"_elbow_joint").c_str());
            d->qpos[m->jnt_qposadr[roll]]=(std::string(side)=="left"?1:-1)*RubikTask::Pi/2;d->qpos[m->jnt_qposadr[elbow]]=RubikTask::Pi/2;
        }
        mj_kinematics(m,d);mj_comPos(m,d);fill(true);
        for(int j=1;j<m->njnt;j++){int actual=mj_name2id(sim.model,mjOBJ_JOINT,mj_id2name(m,mjOBJ_JOINT,j));d->qpos[m->jnt_qposadr[j]]=sim.data->qpos[sim.model->jnt_qposadr[actual]];}
        mj_kinematics(m,d);mj_comPos(m,d);fill(false);
        input.hands[0].position={-.2,1.1,-.4};input.hands[1].position={.2,1.1,-.4};
        int rollJoint=mj_name2id(sim.model,mjOBJ_JOINT,"left_wrist_roll_joint");
        sim.data->qpos[sim.model->jnt_qposadr[rollJoint]]+=.05; // deliberately stale xpos/xmat
        auto* fresh=mj_makeData(sim.model);std::copy_n(sim.data->qpos,sim.model->nq,fresh->qpos);mj_kinematics(sim.model,fresh);
        double freshWrist[4];mju_mat2Quat(freshWrist,fresh->xmat+9*mj_name2id(sim.model,mjOBJ_BODY,"left_wrist_yaw_link"));mj_deleteData(fresh);
        meta.Calibrate(sim.model,sim.data,input);input.xr_time_ns=input.body.time_ns=1000000000;meta.Solve(input);
        auto targets=solver.Targets();Check(RotationError(freshWrist,targets[12].quaternion.data())<1e-9,"Controller calibrated from stale pre-integration FK");auto changed=input;changed.xr_time_ns+=10000000;changed.body.time_ns+=10000000;
        changed.body.joints[12].position[0]+=.2;changed.body.joints[12].quaternion={1,0,0,0};meta.Solve(changed);
        Check(PositionError(targets[12],solver.Targets()[12])<1e-12,"SDK wrist overrode the controller position");
        Check(RotationError(targets[12].quaternion.data(),solver.Targets()[12].quaternion.data())<1e-12,"SDK wrist overrode the controller orientation");
        changed=input;changed.xr_time_ns+=20000000;changed.hands[0].position[0]+=.05;meta.Solve(changed);
        double mapped[3],delta[3]={.05,0,0};mju_mulMatVec(mapped,meta.StageToRobotRotation(),delta,3,3);
        for(int k=0;k<3;k++)Check(std::abs(solver.Targets()[12].position[k]-targets[12].position[k]-.8*mapped[k])<1e-8,"Controller displacement frame/scale");
        Check(PositionError(targets[13],solver.Targets()[13])<1e-12,"Opposite controller moved");
        changed=input;changed.xr_time_ns+=30000000;
        double B[4],inv[4],world[4]={std::cos(.1),0,0,std::sin(.1)},q[4],source[4],expected[4];
        mju_mat2Quat(B,meta.StageToRobotRotation());mju_negQuat(inv,B);mju_mulQuat(q,inv,world);mju_mulQuat(source,q,B);
        mju_mulQuat(q,source,input.hands[0].quaternion.data());mju_copy(changed.hands[0].quaternion.data(),q,4);
        meta.Solve(changed);mju_mulQuat(expected,world,targets[12].quaternion.data());
        Check(RotationError(expected,solver.Targets()[12].quaternion.data())<1e-9,"Controller rotation axis changed");
        changed.hand_active[0]=false;changed.xr_time_ns+=10000000;bool rejected=false;
        try{meta.Solve(changed);}catch(const std::exception&){rejected=true;}
        Check(rejected,"Lost controller treated as a pose");
        meta.EnableControllerStageTargets(true);meta.EnableCameraTracking(true);meta.Calibrate(sim.model,sim.data,input);
        changed=input;changed.xr_time_ns+=100000000;changed.hands[0].position[0]+=.07;changed.hands[0].position[1]+=.09;meta.Solve(changed);
        auto absolute=solver.Targets();
        const auto& cam=meta.CameraCalibrationPose();const auto& head=meta.HeadCalibrationPose();const auto* basis=meta.StageToRobotRotation();
        for(int h=0;h<2;h++){
            double rel[3],projected[3],palm[3];mju_rotVecQuat(palm,meta.ControllerGripOffset(h).data(),absolute[12+h].quaternion.data());
            for(int k=0;k<3;k++)rel[k]=absolute[12+h].position[k]+palm[k]-cam.position[k];mju_mulMatTVec(projected,basis,rel,3,3);
            for(int k=0;k<3;k++)Check(std::abs(head.position[k]+meta.VisualScale()*projected[k]-changed.hands[h].position[k])<1e-9,"Absolute controller target does not match rendered STAGE frame");
        }
        auto distorted=changed;distorted.xr_time_ns+=10000000;
        distorted.body.joints[0].position[0]+=.1;
        distorted.body.joints[12].position[1]-=.3;distorted.body.joints[12].quaternion={0,1,0,0};meta.Pause();meta.Solve(distorted);
        for(int h=0;h<2;h++){
            Check(PositionError(absolute[12+h],solver.Targets()[12+h])<1e-9,"Body/pause changed absolute controller positions");
            Check(RotationError(absolute[12+h].quaternion.data(),solver.Targets()[12+h].quaternion.data())<1e-9,"Body/pause changed controller rotation calibration");
        }
        Check(ControllerPosesValid(distorted),"Valid controllers gated off runtime reference");
        auto lost=distorted;lost.xr_time_ns+=10000000;lost.hand_active[0]=false;
        Check(!ControllerPosesValid(lost),"Inactive controller passed runtime gate");rejected=false;try{meta.Solve(lost);}catch(...){rejected=true;}Check(rejected,"Stage controller tracking loss accepted");
        Check(meta.calibrated,"Controller loss erased rigid calibration");
        auto invalidPose=distorted;invalidPose.location_flags[1]=1;
        Check(!ControllerPosesValid(invalidPose),"Missing controller position passed runtime gate");
        meta.Pause();distorted.xr_time_ns+=20000000;meta.Solve(distorted);
        Check(RotationError(absolute[12].quaternion.data(),solver.Targets()[12].quaternion.data())<1e-9,"Controller recovery changed wrist orientation");
        // Calibrating in a raised source pose and then assuming the same pose
        // as a down-arm calibration must produce the same hand orientation.
        MetaRetargeter down(argv[1]),up(argv[1]),physicalVariant(argv[1]),oldDown(argv[1]),oldUp(argv[1]);
        for(auto* candidate:{&down,&up,&physicalVariant,&oldDown,&oldUp}){
            candidate->EnableControllerStageTargets(true);candidate->EnableCameraTracking(true);
        }
        for(auto* candidate:{&down,&up,&physicalVariant})candidate->EnableControllerAnatomicalFrame(true);
        auto raised=input;double raise[4]={std::cos(.8),std::sin(.8),0,0};
        for(int h=0;h<2;h++){
            double q[4];mju_mulQuat(q,raise,input.hands[h].quaternion.data());mju_copy(raised.hands[h].quaternion.data(),q,4);
            mju_mulQuat(q,raise,input.body.joints[10+h].quaternion.data());mju_copy(raised.body.joints[10+h].quaternion.data(),q,4);
            raised.hands[h].position[1]+=.6;raised.body.joints[12+h].position[1]+=.6;
        }
        down.Calibrate(sim.model,sim.data,input);up.Calibrate(sim.model,sim.data,raised);
        oldDown.Calibrate(sim.model,sim.data,input);oldUp.Calibrate(sim.model,sim.data,raised);
        sim.data->qpos[sim.model->jnt_qposadr[rollJoint]]+=.8;
        physicalVariant.Calibrate(sim.model,sim.data,input);
        auto common=input;common.xr_time_ns=common.body.time_ns=5000000000LL;
        for(auto* candidate:{&down,&up,&physicalVariant,&oldDown,&oldUp})candidate->Solve(common);
        double oldPoseError=0;
        for(int h=0;h<2;h++){
            const auto* expected=down.solver().Targets()[12+h].quaternion.data();
            Check(RotationError(expected,up.solver().Targets()[12+h].quaternion.data())<1e-9,"Raised-arm calibration changes subsequent hand frame");
            Check(RotationError(expected,physicalVariant.solver().Targets()[12+h].quaternion.data())<1e-9,"Physical wrist pose leaked into anatomical calibration");
            oldPoseError=std::max(oldPoseError,RotationError(oldDown.solver().Targets()[12+h].quaternion.data(),oldUp.solver().Targets()[12+h].quaternion.data()));
        }
        Check(oldPoseError>1,"Fixture did not reproduce legacy raised-arm bug");
        std::cout<<"Raised/down calibration invariant; legacy discrepancy_rad="<<oldPoseError<<"\n";
        mj_deleteData(d);std::cout<<"Controller wrist position, rotation, independence and loss checks passed\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

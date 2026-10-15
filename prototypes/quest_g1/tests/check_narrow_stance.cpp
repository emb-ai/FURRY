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
        meta.Calibrate(sim.model,sim.data,input);
        for(double angle : {0., .7, -1.5, 3.0}) {
            auto f=input; f.body.time_ns=f.xr_time_ns=1000000000;
            double turn[4]={std::cos(angle/2),0,std::sin(angle/2),0},q[4];
            mju_mulQuat(q,turn,input.body.joints[0].quaternion.data());
            std::copy_n(q,4,f.body.joints[0].quaternion.begin());
            meta.EnableNarrowStance(false);meta.Pause();meta.Solve(f);auto normal=solver.Targets();
            meta.EnableNarrowStance(true);meta.Pause();meta.Solve(f);auto narrow=solver.Targets();
            double R[9];mju_quat2Mat(R,normal[0].quaternion.data());double yaw=std::atan2(R[3],R[0]);
            double lateral[3]={-std::sin(yaw),std::cos(yaw),0};
            for(int i=0;i<14;i++)for(int k=0;k<3;k++){
                double expected=(i==6?-1:i==7?1:0)*.06*lateral[k];
                Check(std::abs(narrow[i].position[k]-normal[i].position[k]-expected)<1e-10,"Foot-only XY shift or pelvis axis mismatch");
            }
            for(int i=0;i<14;i++)Check(narrow[i].quaternion==normal[i].quaternion,"Target orientation changed");
            for(int j=0;j<m->nq;j++)Check(std::isfinite(solver.data()->qpos[j]),"Nonfinite GMR result");
            for(int j=1;j<m->njnt;j++)if(m->jnt_limited[j]){
                double v=solver.data()->qpos[m->jnt_qposadr[j]];
                Check(v>=m->jnt_range[2*j]-1e-6&&v<=m->jnt_range[2*j+1]+1e-6,"GMR limit violated");
            }
            meta.EnableNarrowStance(false);meta.Pause();meta.Solve(f);auto restored=solver.Targets();
            for(int i=0;i<14;i++)Check(restored[i].position==normal[i].position&&restored[i].quaternion==normal[i].quaternion,"Stance adjustment leaked after switching off");
        }
        mj_deleteData(d);std::cout<<"Narrow stance target/frame/limits/switch-off checks passed\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}

#include "retarget.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

ArmRetargeter::ArmRetargeter(const mjModel* m):model(m),ik(mj_makeData(m)),jacp(3*m->nv),jacr(3*m->nv){
    std::vector<std::string> names;
    for(auto side:{"left","right"})for(auto part:{"hip_pitch","hip_roll","hip_yaw","knee","ankle_pitch","ankle_roll"})names.push_back(std::string(side)+"_"+part+"_joint");
    for(auto part:{"yaw","roll","pitch"})names.push_back(std::string("waist_")+part+"_joint");
    for(auto side:{"left","right"})for(auto part:{"shoulder_pitch","shoulder_roll","shoulder_yaw","elbow","wrist_roll","wrist_pitch","wrist_yaw"})names.push_back(std::string(side)+"_"+part+"_joint");
    for(int i=0;i<29;i++){
        joint[i]=mj_name2id(m,mjOBJ_JOINT,names[i].c_str());
        if(joint[i]<0)throw std::runtime_error("Retarget joint missing");
        qadr[i]=m->jnt_qposadr[joint[i]];vadr[i]=m->jnt_dofadr[joint[i]];
    }
    wrists[0]=mj_name2id(m,mjOBJ_BODY,"left_wrist_yaw_link");
    wrists[1]=mj_name2id(m,mjOBJ_BODY,"right_wrist_yaw_link");
    int pelvis=mj_name2id(m,mjOBJ_BODY,"pelvis");
    int leftArm=mj_name2id(m,mjOBJ_BODY,"left_shoulder_pitch_link");
    int rightArm=mj_name2id(m,mjOBJ_BODY,"right_shoulder_pitch_link");
    bodyGroup.resize(m->nbody,-2);
    for(int b=1;b<m->nbody;b++){
        for(int parent=b;parent>0;parent=m->body_parentid[parent]){
            if(parent==leftArm){bodyGroup[b]=0;break;}
            if(parent==rightArm){bodyGroup[b]=1;break;}
            if(parent==pelvis){bodyGroup[b]=-1;break;}
        }
    }
    if(!ik || wrists[0]<0 || wrists[1]<0)throw std::runtime_error("Retarget model invalid");
}
ArmRetargeter::~ArmRetargeter(){mj_deleteData(ik);}
void ArmRetargeter::Calibrate(const mjData* data,const TrackingFrame& input){
    if(!input.valid)throw std::runtime_error("Both controllers must be tracked to calibrate");
    origin=input;
    // The simulator may own a thread pool; keep the IK scratch allocator
    // independent instead of copying that runtime state with mj_copyData.
    mju_copy(ik->qpos,data->qpos,model->nq);
    if(model->nmocap){
        mju_copy(ik->mocap_pos,data->mocap_pos,3*model->nmocap);
        mju_copy(ik->mocap_quat,data->mocap_quat,4*model->nmocap);
    }
    // Solve in a stationary, upright robot frame; dynamics stay in Simulation.
    ik->qpos[0]=ik->qpos[1]=0;ik->qpos[2]=.793;
    ik->qpos[3]=1;ik->qpos[4]=ik->qpos[5]=ik->qpos[6]=0;
    mj_kinematics(model,ik);mj_comPos(model,ik);
    double headRotation[9];mju_quat2Mat(headRotation,input.head.quaternion.data());
    double fx=-headRotation[2],fz=-headRotation[8],norm=std::hypot(fx,fz);
    if(norm<.2){fx=0;fz=-1;norm=1;}
    fx/=norm;fz/=norm;
    // XR horizontal forward, horizontal left, up -> robot +X,+Y,+Z.
    double b[9]={fx,0,fz, fz,0,-fx, 0,1,0};std::copy_n(b,9,basis);
    for(int h=0;h<2;h++){
        std::copy_n(ik->xpos+3*wrists[h],3,initialPosition[h].begin());
        filteredPosition[h]=initialPosition[h];
        std::copy_n(ik->xmat+9*wrists[h],9,initialRotation[h].begin());
    }
    calibrated=true;
}
std::array<float,29> ArmRetargeter::Solve(const TrackingFrame& input){
    if(!calibrated || !input.valid)throw std::runtime_error("No valid calibrated tracking");
    error_m=0;limited=false;
    for(int h=0;h<2;h++){
        // Standing mode: anchor hand translation to STAGE at calibration.
        // HMD bobbing is not torso motion; it must not move stationary hands.
        // Locomotion/room recenter requires a separate body reference or recalibration.
        double delta[3],local[3],target[3];
        for(int a=0;a<3;a++)delta[a]=input.hands[h].position[a]-origin.hands[h].position[a];
        mju_mulMatVec(local,basis,delta,3,3);
        double length=mju_norm3(local),scale=.8*(length>.45?.45/length:1);
        for(int a=0;a<3;a++)target[a]=initialPosition[h][a]+scale*local[a];
        // The standing manipulation mode excludes targets inside the torso/legs.
        // Smooth Cartesian discontinuities before solving, not just joint commands.
        double projected[3]={std::clamp(target[0],.02,.50),
            (h==0?1:-1)*std::clamp((h==0?1:-1)*target[1],.16,.50),
            std::clamp(target[2],.70,1.35)};
        double motion[3];mju_sub3(motion,projected,filteredPosition[h].data());
        double distance=mju_norm3(motion);
        for(int a=0;a<3;a++){
            if(std::abs(projected[a]-target[a])>.001)limited=true;
            filteredPosition[h][a]+=motion[a]*std::min(1.,.006/std::max(distance,1e-9));
            target[a]=filteredPosition[h][a];
        }
        double current[9],start[9],deltaRotation[9],inBasis[9],mapped[9],targetRotation[9];
        mju_quat2Mat(current,input.hands[h].quaternion.data());mju_quat2Mat(start,origin.hands[h].quaternion.data());
        mju_mulMatMatT(deltaRotation,current,start,3,3,3);
        mju_mulMatMat(inBasis,basis,deltaRotation,3,3,3);
        mju_mulMatMatT(mapped,inBasis,basis,3,3,3);
        mju_mulMatMat(targetRotation,mapped,initialRotation[h].data(),3,3,3);
        int offset=15+h*7;
        std::array<double,7> previous;
        for(int k=0;k<7;k++)previous[k]=ik->qpos[qadr[offset+k]];
        // Damped least squares on seven arm joints; position dominates wrist rotation.
        for(int iteration=0;iteration<5;iteration++){
            mj_kinematics(model,ik);mj_comPos(model,ik);
            mj_jacBody(model,ik,jacp.data(),jacr.data(),wrists[h]);
            double error[6]={},J[42]={};
            for(int a=0;a<3;a++)error[a]=target[a]-ik->xpos[3*wrists[h]+a];
            const double* rotation=ik->xmat+9*wrists[h];
            for(int col=0;col<3;col++){
                double a[3]={rotation[col],rotation[3+col],rotation[6+col]};
                double b[3]={targetRotation[col],targetRotation[3+col],targetRotation[6+col]},cross[3];
                mju_cross(cross,a,b);for(int r=0;r<3;r++)error[3+r]+=.5*.12*cross[r];
            }
            for(int r=0;r<3;r++)for(int k=0;k<7;k++){
                J[r*7+k]=jacp[r*model->nv+vadr[offset+k]];
                J[(r+3)*7+k]=.12*jacr[r*model->nv+vadr[offset+k]];
            }
            double system[36]={},solution[6]={};mju_mulMatMatT(system,J,J,6,7,6);
            for(int r=0;r<6;r++)system[r*6+r]+=.0025;
            mju_cholFactor(system,6,1e-10);mju_cholSolve(solution,system,error,6);
            for(int k=0;k<7;k++){
                double step=0;for(int r=0;r<6;r++)step+=J[r*7+k]*solution[r];
                int i=offset+k;
                ik->qpos[qadr[i]]=std::clamp(ik->qpos[qadr[i]]+std::clamp(step,-.035,.035),model->jnt_range[2*joint[i]]+.025,model->jnt_range[2*joint[i]+1]-.025);
            }
        }
        mj_kinematics(model,ik);
        double difference[3];mju_sub3(difference,target,ik->xpos+3*wrists[h]);
        double residual=mju_norm3(difference);error_m=std::max(error_m,residual);
        bool collision=false;
        if(std::isfinite(residual) && residual<=.06){
            mj_collision(model,ik);
            for(int c=0;c<ik->ncon;c++){
                const auto& contact=ik->contact[c];
                int a=bodyGroup[model->geom_bodyid[contact.geom[0]]];
                int b=bodyGroup[model->geom_bodyid[contact.geom[1]]];
                if(a>=-1 && b>=-1 && a!=b && (a>=0 || b>=0) && contact.dist<-.0025){collision=true;break;}
            }
        }
        if(!std::isfinite(residual) || residual>.06 || collision){
            // Reject an unsolved pose. Do not let subsequent IK iterations wind up
            // from a failed configuration and feed it into the balance policy.
            for(int k=0;k<7;k++)ik->qpos[qadr[offset+k]]=previous[k];
            limited=true;
        }
    }
    std::array<float,29> result{};for(int i=0;i<29;i++)result[i]=ik->qpos[qadr[i]];return result;
}

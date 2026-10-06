#include "meta_retarget.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

// Independent anatomical fixture: Meta FOOT_ANKLE is represented by the robot
// ankle frame, deliberately NOT the GMR toe task frame. No generated gait.
int main(int argc,char**argv){
    if(argc!=2)return 2;
    GmrRetargeter physics(argv[1]);physics.Reset();physics.data()->qpos[2]=.793;
    const double home[]={-.2,0,0,.4,-.2,0,-.2,0,0,.4,-.2,0,0,0,0,0,.2,0,1.2,0,0,0,0,-.2,0,1.2,0,0,0};
    std::copy_n(home,29,physics.data()->qpos+7);mj_forward(physics.model(),physics.data());
    MetaRetargeter meta(argv[1]);auto& g=meta.solver();auto*m=g.model();auto*d=mj_makeData(m);
    for(int k=0;k<12;k++){
        const char* parts[]={"hip_pitch","hip_roll","hip_yaw","knee","ankle_pitch","ankle_roll"};
        std::string expected=std::string(k<6?"left_":"right_")+parts[k%6]+"_joint";
        if(expected!=mj_id2name(m,mjOBJ_JOINT,k+1) || m->jnt_qposadr[k+1]!=7+k)
            throw std::runtime_error("GMR leg order differs from TWIST2 policy order");
    }
    TrackingFrame input;input.body.valid=input.body.supported=input.valid=true;input.location_flags={15,15,15};input.body.confidence=1;
    input.head.position={0,1.75,0};constexpr double humanScale=1.25;
    auto fill=[&](bool rest){
        auto& poses=rest?input.body.rest:input.body.joints;
        for(int i=0;i<14;i++){
            int b=g.tasks()[i].body;
            if(i==6||i==7)b=mj_name2id(m,mjOBJ_BODY,i==6?"left_ankle_roll_link":"right_ankle_roll_link");
            if(i==8||i==9)b=mj_name2id(m,mjOBJ_BODY,i==8?"left_shoulder_pitch_link":"right_shoulder_pitch_link");
            auto*p=d->xpos+3*b;poses[i].position={-p[1]*humanScale,p[2]*humanScale,-p[0]*humanScale};
            double xr[9]={0,-1,0,0,0,1,-1,0,0},r[9];mju_mulMatMat(r,xr,d->xmat+9*g.tasks()[i].body,3,3,3);mju_mat2Quat(poses[i].quaternion.data(),r);input.body.flags[i]=15;
        }
    };
    mj_resetData(m,d);d->qpos[2]=.8;
    for(auto side:{"left","right"}){
        auto joint=[&](const char* name){return m->jnt_qposadr[mj_name2id(m,mjOBJ_JOINT,(std::string(side)+name).c_str())];};
        d->qpos[joint("_shoulder_roll_joint")]=(side==std::string("left")?1:-1)*1.5707963267948966;
        d->qpos[joint("_elbow_joint")]=1.5707963267948966;
    }
    mj_forward(m,d);fill(true);
    // A neutral human spine need not be a vertical line. A 4 cm chest depth
    // offset must not rotate gravity or introduce a false lean into the legs.
    // This is independent of the robot-generated limb fixture above.
    input.body.rest[1].position[2]+=.04;
    double restAnkle=d->xpos[3*mj_name2id(m,mjOBJ_BODY,"right_ankle_roll_link")+2];
    for(int j=1;j<m->njnt;j++){int source=mj_name2id(physics.model(),mjOBJ_JOINT,mj_id2name(m,mjOBJ_JOINT,j));d->qpos[m->jnt_qposadr[j]]=physics.data()->qpos[physics.model()->jnt_qposadr[source]];}
    auto ground=[&]{mj_forward(m,d);double ankle=std::min(d->xpos[3*mj_name2id(m,mjOBJ_BODY,"left_ankle_roll_link")+2],d->xpos[3*mj_name2id(m,mjOBJ_BODY,"right_ankle_roll_link")+2]);d->qpos[2]+=restAnkle-ankle;mj_forward(m,d);};
    ground();fill(false);meta.Calibrate(physics.model(),physics.data(),input);
    std::array<double,36> base;std::copy_n(d->qpos,36,base.begin());
    double maxJointError=0,maxRootError=0,translationError=0;
    for(int n=0;n<501;n++){
        std::copy(base.begin(),base.end(),d->qpos);
        // Lift and place left then right leg, including ankle pitch and roll.
        double phase=(n%250)/250.,lift=.5*(1-std::cos(phase*2*3.141592653589793));const char* side=n<250?"left":"right";
        for(auto delta:{std::pair{"_hip_pitch_joint",-.25*lift},std::pair{"_knee_joint",.5*lift},std::pair{"_ankle_pitch_joint",-.25*lift},std::pair{"_ankle_roll_joint",.08*lift}}){
            int j=mj_name2id(m,mjOBJ_JOINT,(std::string(side)+delta.first).c_str());d->qpos[m->jnt_qposadr[j]]+=delta.second;
        }
        d->qpos[0]=.4*n/500.;ground();fill(false);input.head.position[2]=-d->qpos[0]*humanScale;
        input.body.time_ns=input.xr_time_ns=1000000000LL+n*10000000LL;auto cmd=meta.Solve(input);
        if(n>30){for(int k=0;k<12;k++)maxJointError=std::max(maxJointError,std::abs(double(cmd[6+k])-d->qpos[7+k]));maxRootError=std::max(maxRootError,std::abs(double(cmd[2])-d->qpos[2]));}
        if(n==500)translationError=std::abs(g.data()->qpos[0]-.4);
        auto duplicate=meta.Solve(input);if(duplicate!=cmd)throw std::runtime_error("Duplicate frame changed command");
        auto older=input;older.body.time_ns-=10000000LL;
        if(meta.Solve(older)!=cmd)throw std::runtime_error("Old body frame changed command");
    }
    std::printf("leg mapping: max joint error=%.6f rad, height error=%.6f m, 0.4m translation error=%.6f m\n",maxJointError,maxRootError,translationError);
    meta.Pause();input.body.time_ns+=1000000000LL;
    auto resumed=meta.Solve(input);
    if(resumed[0]!=0 || resumed[1]!=0 || resumed[5]!=0)throw std::runtime_error("Resume differentiated across pause");
    mj_deleteData(d);
    return maxJointError<.03 && maxRootError<.003 && translationError<.002?0:1;
}

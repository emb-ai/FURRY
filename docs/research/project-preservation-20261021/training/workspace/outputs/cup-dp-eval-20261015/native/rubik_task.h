#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <mujoco/mujoco.h>
#include <stdexcept>
#include <string>
#include <deque>

// Internal retaining mechanism only. There is no hand weld, face motor,
// cubie qpos overwrite, or visual-only sticker permutation during play.
// Bindings are reconstructed from userdata so mj_copyData snapshots work.
class RubikTask {
public:
    static constexpr double Pi=3.141592653589793;
    static constexpr int Size=64, Active=0, Start=1, Turns=2, LastFace=3, LastSign=4, Candidate=5, CandidateTime=6, RotationIntent=7, Corners=8;
    inline static constexpr const char* Faces[6]={"R","L","B","F","U","D"};
    inline static constexpr int Axis[6][3]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    int core=-1,initial=-1;
    bool requireRotationIntent=false;
    bool rotationIntent=false;
    double intentTime=-1;
    std::deque<std::pair<double,std::array<double,4>>> intentHistory;
    // Infer a request from operator target orientations, never from a wrist
    // shaken by contact forces or from an approximate IK solution. Common
    // rotation of both hands cancels in their relative quaternion.
    void ObserveGoal(const double* left,const double* right,double time){
        if(core<0)return;
        if(!intentHistory.empty() && time<=intentHistory.back().first)return;
        if(!intentHistory.empty() && time-intentHistory.back().first>.1)intentHistory.clear();
        double inverse[4];std::array<double,4> relative;
        mju_negQuat(inverse,left);
        mju_mulQuat(relative.data(),inverse,right);
        intentHistory.push_back({time,relative});
        while(intentHistory.size()>1 && intentHistory[1].first<=time-.2)intentHistory.pop_front();
        double dot=std::abs(mju_dot(relative.data(),intentHistory.front().second.data(),4));
        rotationIntent=time-intentHistory.front().first>=.18 && 2*std::acos(std::clamp(dot,0.,1.))>.25*(time-intentHistory.front().first);
        intentTime=time;
    }
    bool HasRotationIntent(double time)const{
        return !requireRotationIntent || (rotationIntent && time-intentTime<.05);
    }

    std::array<int,6> qa{},va{};
    std::array<int,8> cubie{},rest{};
    std::array<std::array<int,8>,6> layer{};
    std::array<int,2> contactMasks{};
    std::array<int,2> wrist{-1,-1};
    double cornerSpacing=.024,magnetStiffness=.24,magnetWindow=15*Pi/180,magnetDamping=.00012;
    void Initialize(const mjModel* m){
        core=mj_name2id(m,mjOBJ_BODY,"rubik_core");if(core<0)return;
        requireRotationIntent=mj_name2id(m,mjOBJ_NUMERIC,"rubik_rotation_intent")>=0;
        wrist={mj_name2id(m,mjOBJ_BODY,"left_wrist_yaw_link"),mj_name2id(m,mjOBJ_BODY,"right_wrist_yaw_link")};
        initial=mj_name2id(m,mjOBJ_NUMERIC,"rubik_initial");
        if(m->nuserdata!=Size||initial<0||m->numeric_size[initial]!=56)throw std::runtime_error("Invalid Rubik scene");
        cornerSpacing=std::abs(m->numeric_data[m->numeric_adr[initial]]);
        int magnet=mj_name2id(m,mjOBJ_NUMERIC,"rubik_magnet");
        if(magnet<0||m->numeric_size[magnet]!=3)throw std::runtime_error("Missing magnetic configuration");
        const double* config=m->numeric_data+m->numeric_adr[magnet];
        magnetStiffness=config[0];magnetWindow=config[1];magnetDamping=config[2];
        if(magnetStiffness<=0||magnetWindow<10*Pi/180-1e-8||magnetWindow>15*Pi/180+1e-8||magnetDamping<0)throw std::runtime_error("Invalid magnetic configuration");
        for(int f=0;f<6;f++){
            int j=mj_name2id(m,mjOBJ_JOINT,(std::string("rubik_hinge_")+Faces[f]).c_str());
            if(j<0)throw std::runtime_error("Missing Rubik hinge");qa[f]=m->jnt_qposadr[j];va[f]=m->jnt_dofadr[j];
        }
        for(int i=0;i<8;i++){
            cubie[i]=mj_name2id(m,mjOBJ_BODY,("rubik_cubie_"+std::to_string(i)).c_str());
            rest[i]=mj_name2id(m,mjOBJ_EQUALITY,("rubik_core_weld_"+std::to_string(i)).c_str());
            for(int f=0;f<6;f++)layer[f][i]=mj_name2id(m,mjOBJ_EQUALITY,(std::string("rubik_layer_weld_")+Faces[f]+"_"+std::to_string(i)).c_str());
            if(cubie[i]<0||rest[i]<0)throw std::runtime_error("Missing Rubik cubie");
            for(int f=0;f<6;f++)if(layer[f][i]<0)throw std::runtime_error("Missing Rubik retaining weld");
        }
    }
    void Reset(mjModel* m,mjData* d){
        if(core<0)return;
        intentHistory.clear();rotationIntent=false;intentTime=-1;
        mju_zero(d->userdata,Size);d->userdata[Active]=-1;d->userdata[LastFace]=-1;d->userdata[Candidate]=-1;
        mju_copy(d->userdata+Corners,m->numeric_data+m->numeric_adr[initial],56);
        for(int i=0;i<8;i++)mju_normalize4(d->userdata+Corners+i*7+3);
        Configure(m,d);mj_forward(m,d);
    }
    static void Rotation(double* q,int face,double angle){
        q[0]=std::cos(angle/2);for(int k=0;k<3;k++)q[1+k]=Axis[face][k]*std::sin(angle/2);
    }
    static bool OnFace(const double* p,int f){double dot=0;for(int k=0;k<3;k++)dot+=p[k]*Axis[f][k];return dot>.008;}
    static void Binding(mjModel* m,int eq,const double* p,const double* q){
        mju_copy(m->eq_data+eq*mjNEQDATA+3,p,3);mju_copy(m->eq_data+eq*mjNEQDATA+6,q,4);
    }
    void Configure(mjModel* m,mjData* d)const{
        if(core<0)return;
        int f=int(d->userdata[Active]);if(f< -1||f>=6)throw std::runtime_error("Corrupt Rubik layer state");
        double inverse[4]={1,0,0,0};if(f>=0)Rotation(inverse,f,-d->userdata[Start]);
        for(int i=0;i<8;i++){
            const double* state=d->userdata+Corners+i*7;
            Binding(m,rest[i],state,state+3);
            bool selected=f>=0&&OnFace(state,f);
            d->eq_active[rest[i]]=!selected;
            for(int k=0;k<6;k++)d->eq_active[layer[k][i]]=selected&&k==f;
            if(selected){
                double p[3],q[4];mju_rotVecQuat(p,state,inverse);mju_mulQuat(q,inverse,state+3);
                Binding(m,layer[f][i],p,q);
            }
        }
    }
    bool Activate(mjModel* m,mjData* d,int face){
        if(core<0||face<0||face>=6)return false;
        int previous=int(d->userdata[Active]);
        if(previous>=0&&(std::abs(d->qpos[qa[previous]]-d->userdata[Start])>.035||std::abs(d->qvel[va[previous]])>.5))return false;
        d->userdata[Active]=face;d->userdata[Start]=d->qpos[qa[face]];
        Configure(m,d);return true;
    }
    static double MagneticTorque(double angle,double speed,double stiffness=.24,double window=15*Pi/180,double damping=.00012){
        double offset=std::remainder(angle,Pi/2),weight=std::max(0.,1-std::abs(offset)/window);
        return -stiffness*offset*weight*weight-damping*speed;
    }
    int TorqueFace(const mjModel* m,const mjData* d)const{
        double scores[6]={};
        const double* matrix=d->xmat+core*9;
        for(int c=0;c<d->ncon;c++){
            const auto& contact=d->contact[c];int cb=-1,side=-1;
            int ids[2]={m->geom_bodyid[contact.geom1],m->geom_bodyid[contact.geom2]};
            for(int i=0;i<8;i++)for(int s=0;s<2;s++)if(ids[s]==cubie[i]){cb=i;side=s;}
            if(cb<0)continue;
            const char* name=mj_id2name(m,mjOBJ_BODY,ids[1-side]);
            if(!name||std::string(name).find("_hand_")==std::string::npos)continue;
            int hand=std::string(name).rfind("left",0)==0?0:1;
            if(wrist[hand]<0)continue;
            double contactForce[6],handVelocity[6],counterVelocity[6];
            mj_contactForce(m,d,c,contactForce);if(contactForce[0]<.08)continue;
            mj_objectVelocity(m,d,mjOBJ_BODY,wrist[hand],handVelocity,0);
            if(wrist[1-hand]<0)continue;
            mj_objectVelocity(m,d,mjOBJ_BODY,wrist[1-hand],counterVelocity,0);
            double relativeRotation[3];
            for(int k=0;k<3;k++)relativeRotation[k]=handVelocity[k]-counterVelocity[k];
            for(int f=0;f<6;f++){
                if(!OnFace(d->userdata+Corners+cb*7,f))continue;
                bool counterHold=false;
                for(int i=0;i<8;i++)if((contactMasks[1-hand]&(1<<i))&&!OnFace(d->userdata+Corners+i*7,f))counterHold=true;
                if(!counterHold)continue;
                double axis[3]={};
                for(int k=0;k<3;k++)for(int j=0;j<3;j++)axis[k]+=matrix[k*3+j]*Axis[f][j];
                double rate=std::abs(mju_dot3(relativeRotation,axis));
                if(rate>.25)scores[f]+=contactForce[0]*rate;
            }
        }
        int best=-1;double maximum=.01;
        for(int f=0;f<6;f++)if(scores[f]>maximum){maximum=scores[f];best=f;}
        return best;
    }
    std::array<int,2> Contacts(const mjModel* m,const mjData* d)const{
        std::array<int,2> masks{};
        for(int c=0;c<d->ncon;c++){
            double force[6];mj_contactForce(m,d,c,force);if(force[0]<.08)continue;
            const auto& contact=d->contact[c];int body[2]={m->geom_bodyid[contact.geom1],m->geom_bodyid[contact.geom2]};
            for(int side=0;side<2;side++){
                const char* n=mj_id2name(m,mjOBJ_BODY,body[side]);if(!n)continue;std::string name(n);
                int hand=name.rfind("left_hand_",0)==0||name=="left_wrist_yaw_link"?0:
                    name.rfind("right_hand_",0)==0||name=="right_wrist_yaw_link"?1:-1;
                if(hand<0)continue;
                for(int i=0;i<8;i++)if(body[1-side]==cubie[i])masks[hand]|=1<<i;
            }
        }
        return masks;
    }
    void Begin(mjModel* m,mjData* d,double leftGrip,double rightGrip){
        if(core<0)return;
        (void)leftGrip;(void)rightGrip;
        contactMasks=Contacts(m,d);
        int requested=(!requireRotationIntent||d->userdata[RotationIntent]>.5)?TorqueFace(m,d):-1;
        if(requested!=int(d->userdata[Candidate])){
            d->userdata[Candidate]=requested;d->userdata[CandidateTime]=0;
        }else if(requested>=0)d->userdata[CandidateTime]+=m->opt.timestep;
        if(requested>=0&&requested!=int(d->userdata[Active])&&d->userdata[CandidateTime]>=.04)Activate(m,d,requested);
        Configure(m,d);
        for(int f=0;f<6;f++){
            double angle=d->qpos[qa[f]],speed=d->qvel[va[f]];
            d->qfrc_applied[va[f]]=f==int(d->userdata[Active])?MagneticTorque(angle-d->userdata[Start],speed,magnetStiffness,magnetWindow,magnetDamping):0;
        }
    }
    void Finish(mjModel* m,mjData* d){
        if(core<0)return;int f=int(d->userdata[Active]);if(f<0)return;
        double delta=d->qpos[qa[f]]-d->userdata[Start],speed=d->qvel[va[f]];
        int quarter=int(std::round(delta/(Pi/2)));
        (void)speed;
        if(quarter==0)return;
        if(std::abs(delta-quarter*Pi/2)>magnetWindow)return;
        if(std::abs(quarter)>1)throw std::runtime_error("Rubik integration skipped a quarter revolution");
        double rotation[4];Rotation(rotation,f,quarter*Pi/2);
        for(int i=0;i<8;i++){
            double* state=d->userdata+Corners+i*7;if(!OnFace(state,f))continue;
            double p[3],q[4];mju_rotVecQuat(p,state,rotation);mju_mulQuat(q,rotation,state+3);
            for(int k=0;k<3;k++)state[k]=std::round(p[k]/cornerSpacing)*cornerSpacing;
            mju_normalize4(q);mju_copy(state+3,q,4);
        }
        d->userdata[Turns]++;d->userdata[LastFace]=f;d->userdata[LastSign]=quarter;
        // Coordinate relabel only: keep the hinge and all physical qpos/qvel.
        // The layer stays free to wobble around its magnetic equilibrium.
        d->userdata[Start]+=quarter*Pi/2;
        Configure(m,d);
    }
    bool Solved(const mjData* d)const{
        if(core<0)return false;
        int active=int(d->userdata[Active]);
        if(active>=0&&std::abs(d->qpos[qa[active]]-d->userdata[Start])>.026)return false;
        std::array<int,6> colors;colors.fill(-1);
        for(int i=0;i<8;i++)for(int axis=0;axis<3;axis++){
            double sticker[3]={},normal[3];int sign=(i&(1<<(2-axis)))?1:-1;
            sticker[axis]=sign;mju_rotVecQuat(normal,sticker,d->userdata+Corners+i*7+3);
            int direction=0;for(int k=1;k<3;k++)if(std::abs(normal[k])>std::abs(normal[direction]))direction=k;
            int face=2*direction+(normal[direction]<0),color=2*axis+(sign<0);
            if(colors[face]<0)colors[face]=color;else if(colors[face]!=color)return false;
        }
        return true;
    }
    std::string Status(const mjData* d)const{
        if(core<0)return {};char text[160];int face=int(d->userdata[Active]);
        std::snprintf(text,sizeof(text),"2x2: %s / turns %d%s%s",Solved(d)?"SOLVED":"scrambled",int(d->userdata[Turns]),face>=0?" / layer ":"",face>=0?Faces[face]:"");return text;
    }
};

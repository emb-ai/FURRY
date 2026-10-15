#pragma once
#include <mujoco/mujoco.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <stdexcept>
#include "cup_placement_cue.h"

// Versioned collection scenes: deterministic starts and persistent trial state.
// All outcome state is in userdata and recorded; scoring never moves a prop.
class CollectionTask {
public:
    enum {Seed,Elapsed,Settled,Outcome,Lift0,Lift1,StartZ0,StartZ1,StateSize};
    int kind=0,count=0,doorJoint=-1,cupGoalMocap=-1;
    std::array<double,8> cupGoalBounds{};
    bool requireHands=false,basketSides=false;
    double doorX=0,doorHalfWidth=0;
    // For one-object door tasks: Lift1 stores passage; StartZ1 previous root X.
    double timeout=90,fallHeight=.50,fallUpright=.707;
    std::array<int,2> body{-1,-1},joint{-1,-1};
    std::array<std::array<double,6>,2> start{},goal{};
    static uint32_t Next(uint32_t& s){s+=0x9e3779b9u;uint32_t z=s;z=(z^(z>>16))*0x85ebca6bu;z=(z^(z>>13))*0xc2b2ae35u;return z^(z>>16);}
    static double Uniform(uint32_t& s,double lo,double hi){return lo+(hi-lo)*(Next(s)/4294967296.);}
    void Initialize(const mjModel* m){
        int config=mj_name2id(m,mjOBJ_NUMERIC,"collection_task");if(config<0)return;
        const auto* c=m->numeric_data+m->numeric_adr[config];kind=int(c[0]);timeout=c[1];count=kind==3?2:1;
        if(kind<1||kind>4||m->nuserdata!=StateSize)throw std::runtime_error("Invalid collection task");
        int fall=mj_name2id(m,mjOBJ_NUMERIC,"collection_fall");
        if(fall>=0){
            if(m->numeric_size[fall]!=2)throw std::runtime_error("Invalid task fall criterion");
            auto* f=m->numeric_data+m->numeric_adr[fall];fallHeight=f[0];fallUpright=f[1];
            if(!std::isfinite(fallHeight)||fallHeight<=0||!std::isfinite(fallUpright)||fallUpright<=0||fallUpright>=1)
                throw std::runtime_error("Invalid task fall thresholds");
        }
        requireHands=mj_name2id(m,mjOBJ_NUMERIC,"collection_bimanual_hands")>=0;
        basketSides=mj_name2id(m,mjOBJ_NUMERIC,"collection_basket_sides")>=0;
        int door=mj_name2id(m,mjOBJ_NUMERIC,"collection_door");
        int cupBounds=mj_name2id(m,mjOBJ_NUMERIC,"collection_cup_goal_bounds");
        if(cupBounds>=0){
            int marker=mj_name2id(m,mjOBJ_BODY,"cup_goal_marker");
            if(kind!=1||m->numeric_size[cupBounds]!=8||marker<0||m->body_mocapid[marker]<0)
                throw std::runtime_error("Invalid cup goal sampler");
            cupGoalMocap=m->body_mocapid[marker];
            std::copy_n(m->numeric_data+m->numeric_adr[cupBounds],8,cupGoalBounds.begin());
        }
        if(door>=0){
            if(kind!=4||m->numeric_size[door]!=2)throw std::runtime_error("Invalid door task");
            doorJoint=mj_name2id(m,mjOBJ_JOINT,"basket_door_hinge");
            if(doorJoint<0)throw std::runtime_error("Missing door hinge");
            auto* c=m->numeric_data+m->numeric_adr[door];doorX=c[0];doorHalfWidth=c[1];
        }
        for(int i=0;i<count;i++){
            std::string name=kind==1?"cup":"task_object_"+std::to_string(i);
            body[i]=mj_name2id(m,mjOBJ_BODY,name.c_str());joint[i]=mj_name2id(m,mjOBJ_JOINT,(name+"_free").c_str());
            for(auto pair:{std::make_pair("collection_start_",&start[i]),std::make_pair("collection_goal_",&goal[i])}){
                int n=mj_name2id(m,mjOBJ_NUMERIC,(std::string(pair.first)+std::to_string(i)).c_str());
                if(n<0||m->numeric_size[n]!=6||body[i]<0||joint[i]<0)throw std::runtime_error("Incomplete collection task");
                std::copy_n(m->numeric_data+m->numeric_adr[n],6,pair.second->begin());
            }
        }
    }
    void Reset(const mjModel* m,mjData* d,uint32_t seed){
        if(!kind)return;
        std::fill_n(d->userdata,StateSize,0);d->userdata[Seed]=seed;
        uint32_t rng=seed;
        if(cupGoalMocap>=0){
            // Draw the task distance band first, independent of left/right.
            int band=(Next(rng)&1)?6:4;
            bool found=false;
            for(int attempt=0;attempt<4096;attempt++){
                double sx=Uniform(rng,start[0][0],start[0][1]);
                double sy=Uniform(rng,start[0][2],start[0][3]);
                double gx=Uniform(rng,cupGoalBounds[0],cupGoalBounds[1]);
                double gy=Uniform(rng,cupGoalBounds[2],cupGoalBounds[3]);
                double distance=std::hypot(sx-gx,sy-gy);
                if(distance<cupGoalBounds[band]||distance>cupGoalBounds[band+1])continue;
                // Keep the easy half on one side for a direct one-hand move.
                // Long transfers may cross the midline and use either hand.
                if(band==4 && sy*gy<0)continue;
                // Save only the visual target here; sample the object below
                // using the same accepted coordinates, without touching model data.
                auto* target=d->mocap_pos+3*cupGoalMocap;
                target[0]=gx;target[1]=gy;target[2]=goal[0][2];
                int q=m->jnt_qposadr[joint[0]];
                d->qpos[q]=sx;d->qpos[q+1]=sy;found=true;break;
            }
            if(!found)throw std::runtime_error("Unreachable cup layout sampling bounds");
        }
        for(int i=0;i<count;i++){
            int q=m->jnt_qposadr[joint[i]],v=m->jnt_dofadr[joint[i]];auto b=start[i];
            double x,y;
            if(cupGoalMocap>=0){x=d->qpos[q];y=d->qpos[q+1];}
            else do{x=Uniform(rng,b[0],b[1]);y=Uniform(rng,b[2],b[3]);}while(kind==1&&std::hypot(x-goal[i][0],y-goal[i][1])<.15);
            double yaw=Uniform(rng,-b[5],b[5]);
            if(basketSides && (Next(rng)&1))y=-y;
            d->qpos[q]=x;d->qpos[q+1]=y;d->qpos[q+2]=b[4];d->qpos[q+3]=std::cos(yaw/2);
            d->qpos[q+4]=d->qpos[q+5]=0;d->qpos[q+6]=std::sin(yaw/2);std::fill_n(d->qvel+v,6,0);
            d->userdata[StartZ0+i]=b[4];
        }
        if(doorJoint>=0){
            d->qpos[m->jnt_qposadr[doorJoint]]=0;d->qvel[m->jnt_dofadr[doorJoint]]=0;
            d->userdata[StartZ1]=d->qpos[0];
        }
        mj_forward(m,d);
    }
    std::array<double,6> Goal(const mjModel* m,const mjData* d,int i) const{
        auto g=goal[i];
        if(cupGoalMocap>=0&&i==0){auto target=CupGoal(m,d);for(int k=0;k<3;k++)g[k]=target[k];}
        return g;
    }
    void Finish(const mjModel* m,mjData* d){
        if(!kind||d->userdata[Outcome])return;
        auto* s=d->userdata;s[Elapsed]+=m->opt.timestep;
        if(!std::isfinite(d->qpos[2])||d->qpos[2]<fallHeight||d->xmat[9*mj_name2id(m,mjOBJ_BODY,"pelvis")+8]<fallUpright){s[Outcome]=2;return;}
        if(doorJoint>=0){
            const auto* basket=d->xpos+body[0]*3;
            if(s[StartZ1]<doorX && d->qpos[0]>=doorX && std::abs(d->qpos[1])<doorHalfWidth &&
               std::abs(basket[0]-doorX)<.7 && std::abs(basket[1])<doorHalfWidth &&
               basket[2]>s[StartZ0]+.045 && d->qpos[m->jnt_qposadr[doorJoint]]>.5)s[Lift1]=1;
            s[StartZ1]=d->qpos[0];
        }
        bool all=doorJoint<0||s[Lift1];
        for(int i=0;i<count;i++){
            const auto* p=d->xpos+body[i]*3;auto g=Goal(m,d,i);
            mjtNum vel[6];mj_objectVelocity(m,d,mjOBJ_BODY,body[i],vel,0);
            bool touching=false,correctHand=false;
            for(int c=0;c<d->ncon;c++)for(int side=0;side<2;side++){
                int a=d->contact[c].geom[side],b=d->contact[c].geom[1-side];
                if(a>=0&&b>=0&&m->geom_bodyid[a]==body[i]&&(m->geom_contype[b]&4)&&d->contact[c].dist<=0){
                    touching=true;const char* name=mj_id2name(m,mjOBJ_BODY,m->geom_bodyid[b]);
                    if(name && std::string(name).rfind(i==0?"left_":"right_",0)==0)correctHand=true;
                }
            }
            if(p[2]>s[StartZ0+i]+.045 && (!requireHands||correctHand))s[Lift0+i]=1;
            bool inside=std::abs(p[0]-g[0])<g[3]&&std::abs(p[1]-g[1])<g[4]&&std::abs(p[2]-g[2])<g[5];
            if(kind==1)inside=std::hypot(p[0]-g[0],p[1]-g[1])<g[3]&&std::abs(p[2]-g[2])<g[5];
            all=all&&s[Lift0+i]&&inside&&!touching&&mju_norm3(vel+3)<.06&&mju_norm3(vel)<.25;
            if(kind==1||kind==4)all=all&&d->xmat[9*body[i]+8]>.94;
        }
        s[Settled]=all?s[Settled]+m->opt.timestep:0;
        if(s[Settled]>=.5)s[Outcome]=1;
        else if(s[Elapsed]>=timeout)s[Outcome]=3;
    }
    std::string Status(const mjData* d)const{
        if(!kind)return "";
        switch(int(d->userdata[Outcome])){
        case 1:return "Готово: предметы отпущены. Сохраните запись; сброс — новый старт.";
        case 2:return "Падение. Сохраните попытку; сброс — новый старт.";
        case 3:return "Время вышло. Сохраните попытку; сброс — новый старт.";
        }
        const char* titles[]={"","Поднимите чашку и поставьте на круг","Чашку с полки — в коробку на полу","Оранжевый левой, синий правой — в свои лотки","Корзину через дверь — на полку"};
        return titles[kind];
    }
};

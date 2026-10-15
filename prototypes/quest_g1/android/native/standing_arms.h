#pragma once
#include "retarget.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

// Solve only the two arms against the physical torso. GMR may move its
// reference root; a Balance actor cannot reproduce that translation. This
// correction retains the controller wrist poses when the torso sways.
class StandingArms {
    std::array<std::array<int,7>,2> qa{},va{},joint{};
    std::array<int,2> wrist{};
    std::vector<mjtNum> jp,jr;
    bool ready=false;double lastTime=0;std::array<float,14> previous{};
public:
    void Reset(){ready=false;}
    void Initialize(const mjModel* m){
        jp.resize(3*m->nv);jr.resize(3*m->nv);
        int h=0;
        for(auto side:{"left","right"}){
            wrist[h]=mj_name2id(m,mjOBJ_BODY,(std::string(side)+"_wrist_yaw_link").c_str());
            int i=0;
            for(auto part:{"shoulder_pitch","shoulder_roll","shoulder_yaw","elbow","wrist_roll","wrist_pitch","wrist_yaw"}){
                int j=mj_name2id(m,mjOBJ_JOINT,(std::string(side)+"_"+part+"_joint").c_str());
                if(j<0||wrist[h]<0)throw std::runtime_error("Missing standing arm joint");
                joint[h][i]=j;qa[h][i]=m->jnt_qposadr[j];va[h][i++]=m->jnt_dofadr[j];
            }
            h++;
        }
    }
    std::array<float,35> Correct(const mjModel* m,const mjData* actual,mjData* scratch,
                                const std::array<float,35>& source,const std::array<TrackedPose,2>& goals){
        auto out=source;std::copy_n(actual->qpos,m->nq,scratch->qpos);
        if(!ready||actual->time<lastTime){for(int h=0;h<2;h++)for(int i=0;i<7;i++)previous[h*7+i]=actual->qpos[qa[h][i]];lastTime=actual->time;ready=true;}
        const double dt=std::clamp(actual->time-lastTime,0.,.02);lastTime=actual->time;
        for(int h=0;h<2;h++){
            for(int i=0;i<7;i++)scratch->qpos[qa[h][i]]=std::clamp(double(previous[7*h+i]),m->jnt_range[2*joint[h][i]],m->jnt_range[2*joint[h][i]+1]);
            for(int iteration=0;iteration<12;iteration++){
                mj_kinematics(m,scratch);mj_comPos(m,scratch);
                mj_jacBody(m,scratch,jp.data(),jr.data(),wrist[h]);
                mjtNum inv[4],delta[4],angle[3],e[6],J[42];
                mju_negQuat(inv,scratch->xquat+4*wrist[h]);mju_mulQuat(delta,goals[h].quaternion.data(),inv);mju_quat2Vel(angle,delta,1);
                for(int a=0;a<3;a++){
                    e[a]=goals[h].position[a]-scratch->xpos[3*wrist[h]+a];e[3+a]=.12*angle[a];
                    for(int i=0;i<7;i++){J[a*7+i]=jp[a*m->nv+va[h][i]];J[(a+3)*7+i]=.12*jr[a*m->nv+va[h][i]];}
                }
                mjtNum A[49]{},b[7]{},step[7];
                for(int i=0;i<7;i++){
                    // Small posture preference resolves redundancy without
                    // competing materially with centimetre-scale hand motion.
                    A[7*i+i]=.0004;b[i]=.00005*(source[21+7*h+i]-scratch->qpos[qa[h][i]]);
                    for(int a=0;a<6;a++){
                        b[i]+=J[a*7+i]*e[a];
                        for(int j=0;j<7;j++)A[7*i+j]+=J[a*7+i]*J[a*7+j];
                    }
                }
                if(mju_cholFactor(A,7,1e-10)!=7)break;mju_cholSolve(step,A,b,7);
                double scale=1;for(double v:step)scale=std::min(scale,.15/std::max(.15,std::abs(v)));
                for(int i=0;i<7;i++){
                    const int j=joint[h][i];
                    scratch->qpos[qa[h][i]]=std::clamp(scratch->qpos[qa[h][i]]+scale*step[i],m->jnt_range[2*j]+1e-5,m->jnt_range[2*j+1]-1e-5);
                }
            }
            for(int i=0;i<7;i++){
                previous[7*h+i]+=std::clamp(scratch->qpos[qa[h][i]]-previous[7*h+i],-3*dt,3*dt);
                out[21+7*h+i]=previous[7*h+i];
            }
        }
        return out;
    }
};

#pragma once
#include <array>
#include <cmath>
#include <mujoco/mujoco.h>

// Target pose is recorded in scene_state.csv mocap values for randomized cups.
inline std::array<double,4> CupGoal(const mjModel* m,const mjData* d){
    int id=mj_name2id(m,mjOBJ_NUMERIC,"cup_lab_goal");
    if(id<0||m->numeric_size[id]!=4)return {};
    const auto* v=m->numeric_data+m->numeric_adr[id];
    std::array<double,4> result{v[0],v[1],v[2],v[3]};
    int body=mj_name2id(m,mjOBJ_BODY,"cup_goal_marker");
    if(body>=0&&m->body_mocapid[body]>=0)
        for(int k=0;k<3;k++)result[k]=d->mocap_pos[3*m->body_mocapid[body]+k];
    return result;
}

// Visual placement feedback, separate from the historical pickup task score.
// Only read the physics-worker's data; publish the result with its geometry.
inline std::array<float,4> CupPlacementCue(const mjModel* m,const mjData* d){
    int cup=mj_name2id(m,mjOBJ_BODY,"cup"),goal=mj_name2id(m,mjOBJ_NUMERIC,"cup_lab_goal");
    if(cup<0||goal<0||m->numeric_size[goal]!=4)return {};
    const auto* p=d->xpos+cup*3;
    const auto target=CupGoal(m,d);
    mjtNum velocity[6];mj_objectVelocity(m,d,mjOBJ_BODY,cup,velocity,0);
    bool settled=std::hypot(p[0]-target[0],p[1]-target[1])<target[3]&&
        std::abs(p[2]-target[2])<.012&&d->xmat[cup*9+8]>.966&&
        mju_norm3(velocity+3)<.05&&mju_norm3(velocity)<.2;
    return {float(p[0]),float(p[1]),float(p[2]),settled?1.f:0.f};
}

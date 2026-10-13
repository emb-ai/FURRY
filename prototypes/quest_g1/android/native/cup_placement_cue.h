#pragma once
#include <array>
#include <cmath>
#include <mujoco/mujoco.h>

// Visual placement feedback, separate from the historical pickup task score.
// Only read the physics-worker's data; publish the result with its geometry.
inline std::array<float,4> CupPlacementCue(const mjModel* m,const mjData* d){
    int cup=mj_name2id(m,mjOBJ_BODY,"cup"),goal=mj_name2id(m,mjOBJ_NUMERIC,"cup_lab_goal");
    if(cup<0||goal<0||m->numeric_size[goal]!=4)return {};
    const auto* p=d->xpos+cup*3;
    const auto* target=m->numeric_data+m->numeric_adr[goal];
    mjtNum velocity[6];mj_objectVelocity(m,d,mjOBJ_BODY,cup,velocity,0);
    bool settled=std::hypot(p[0]-target[0],p[1]-target[1])<target[3]&&
        std::abs(p[2]-target[2])<.012&&d->xmat[cup*9+8]>.966&&
        mju_norm3(velocity+3)<.05&&mju_norm3(velocity)<.2;
    return {float(p[0]),float(p[1]),float(p[2]),settled?1.f:0.f};
}

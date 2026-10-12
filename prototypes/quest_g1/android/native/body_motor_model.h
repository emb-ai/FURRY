#pragma once
#include "body_motor_profile.h"
#include <mujoco/mujoco.h>

namespace bodymotors {
inline Motor Capture(const mjModel* model,const std::string& name) {
    const int j=mj_name2id(model,mjOBJ_JOINT,name.c_str()),a=mj_name2id(model,mjOBJ_ACTUATOR,name.c_str());
    if(j<0||a<0||model->jnt_type[j]!=mjJNT_HINGE||model->actuator_trnid[2*a]!=j)
        throw std::runtime_error("Body motor mapping differs: "+name);
    const int v=model->jnt_dofadr[j];Motor out;out.name=name;
    out.armature=model->dof_armature[v];out.damping=model->dof_damping[v];out.friction=model->dof_frictionloss[v];
    for(int k=0;k<2;k++){
        out.range[k]=model->jnt_range[2*j+k];out.jointForce[k]=model->jnt_actfrcrange[2*j+k];
        out.control[k]=model->actuator_ctrlrange[2*a+k];out.actuatorForce[k]=model->actuator_forcerange[2*a+k];
    }
    out.jointLimited=model->jnt_limited[j];out.jointForceLimited=model->jnt_actfrclimited[j];
    out.controlLimited=model->actuator_ctrllimited[a];out.actuatorForceLimited=model->actuator_forcelimited[a];
    return out;
}
inline void ValidateMapping(const mjModel* model,const Profile& profile) {
    for(const auto& motor:profile)Capture(model,motor.name);
}
inline void Apply(mjModel* model,const Profile& profile) {
    // The complete mapping is validated before any model mutation.
    ValidateMapping(model,profile);
    for(const auto& motor:profile){
        const int j=mj_name2id(model,mjOBJ_JOINT,motor.name.c_str()),a=mj_name2id(model,mjOBJ_ACTUATOR,motor.name.c_str()),v=model->jnt_dofadr[j];
        model->dof_armature[v]=motor.armature;model->dof_damping[v]=motor.damping;model->dof_frictionloss[v]=motor.friction;
        for(int k=0;k<2;k++){
            model->jnt_range[2*j+k]=motor.range[k];model->jnt_actfrcrange[2*j+k]=motor.jointForce[k];
            model->actuator_ctrlrange[2*a+k]=motor.control[k];model->actuator_forcerange[2*a+k]=motor.actuatorForce[k];
        }
        model->jnt_limited[j]=motor.jointLimited;model->jnt_actfrclimited[j]=motor.jointForceLimited;
        model->actuator_ctrllimited[a]=motor.controlLimited;model->actuator_forcelimited[a]=motor.actuatorForceLimited;
    }
}
inline void ValidateApplied(const mjModel* model,const Profile& profile) {
    for(const auto& expected:profile){
        const auto actual=Capture(model,expected.name);
        if(actual.armature!=expected.armature||actual.damping!=expected.damping||actual.friction!=expected.friction||
           actual.range!=expected.range||actual.jointForce!=expected.jointForce||actual.control!=expected.control||
           actual.actuatorForce!=expected.actuatorForce||actual.jointLimited!=expected.jointLimited||
           actual.jointForceLimited!=expected.jointForceLimited||actual.controlLimited!=expected.controlLimited||
           actual.actuatorForceLimited!=expected.actuatorForceLimited)
            throw std::runtime_error("Active body motor profile differs: "+expected.name);
    }
}
} // namespace bodymotors

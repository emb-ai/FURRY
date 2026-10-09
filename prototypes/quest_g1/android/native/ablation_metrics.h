#pragma once
#include "simulation.h"
#include "gmr.h"
#include "meta_retarget.h"
#include <array>
#include <ostream>
#include <cmath>
#include <limits>

// Offline diagnostics only. Compact, consistently ordered robot state even
// when the physics model also contains fingers and free objects.
inline void AblationHeader(std::ostream& out) {
    out << ",segment,applying,contacts,foot_force,foot_slip,ref_available";
    for (auto field : {"q", "ref"}) for(int k=0;k<36;k++) out<<','<<field<<k;
    for(int k=0;k<35;k++) out<<",v"<<k;
    // Per-foot floor contact for trip analysis. Braking is the horizontal
    // floor force opposing the contact point's travel; toe_target is the GMR
    // toe-task height relative to a flat grounded foot (NaN without GMR).
    for (auto field : {"floor_force", "floor_brake", "toe_target"}) for (auto side : {"_l", "_r"}) out<<','<<field<<side;
}
// Toe-task targets of the current Meta solve, relative to the flat-foot level.
// Negative values ask GMR to put the toe frame below the grounded sole.
inline std::array<double,2> ToeTargetClearance(MetaRetargeter& meta) {
    const auto& targets=meta.solver().Targets();
    if(!meta.calibrated || targets.size()<8)
        return {std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN()};
    return {targets[6].position[2]-meta.FlatToeHeight(0),targets[7].position[2]-meta.FlatToeHeight(1)};
}
inline void AblationRow(std::ostream& out, Simulation& sim, GmrRetargeter& g,
                        int segment, bool applying, bool referenceAvailable=true,
                        std::array<double,2> toeTarget={std::numeric_limits<double>::quiet_NaN(),
                                                        std::numeric_limits<double>::quiet_NaN()}) {
    double forceSum=0, slipSum=0, floorForce[2]={}, floorBrake[2]={};
    for(int c=0;c<sim.data->ncon;c++) {
        const auto& contact=sim.data->contact[c];
        int a=sim.model->geom_bodyid[contact.geom1],b=sim.model->geom_bodyid[contact.geom2];
        int foot=a==0?b:(b==0?a:-1);
        if(foot<0)continue;
        const char* name=mj_id2name(sim.model,mjOBJ_BODY,foot);
        if(!name || (std::string(name)!="left_ankle_roll_link" && std::string(name)!="right_ankle_roll_link"))continue;
        int side=std::string(name)=="left_ankle_roll_link"?0:1;
        double force[6],velocity[3],world[3];std::vector<double> jp(3*sim.model->nv);
        mj_contactForce(sim.model,sim.data,c,force);
        if(force[0]<=1)continue;
        mj_jac(sim.model,sim.data,jp.data(),nullptr,contact.pos,foot);
        mju_mulMatVec(velocity,jp.data(),sim.data->qvel,3,sim.model->nv);
        forceSum+=force[0];slipSum+=force[0]*std::hypot(velocity[0],velocity[1]);
        floorForce[side]+=force[0];
        // The contact-frame force acts on geom2; flip it when the foot is geom1.
        mju_mulMatTVec(world,contact.frame,force,3,3);
        if(foot==a)mju_scl3(world,world,-1);
        double speed=std::hypot(velocity[0],velocity[1]);
        if(speed>.05)floorBrake[side]-=(world[0]*velocity[0]+world[1]*velocity[1])/speed;
    }
    out<<','<<segment<<','<<applying<<','<<sim.data->ncon<<','<<forceSum<<','<<(forceSum?slipSum/forceSum:0)<<','<<referenceAvailable;
    for(int k=0;k<7;k++)out<<','<<sim.data->qpos[k];
    for(int j=1;j<g.model()->njnt;j++){
        int s=mj_name2id(sim.model,mjOBJ_JOINT,mj_id2name(g.model(),mjOBJ_JOINT,j));
        out<<','<<sim.data->qpos[sim.model->jnt_qposadr[s]];
    }
    for(int k=0;k<36;k++)out<<','<<(referenceAvailable?g.data()->qpos[k]:std::numeric_limits<double>::quiet_NaN());
    for(int k=0;k<6;k++)out<<','<<sim.data->qvel[k];
    for(int j=1;j<g.model()->njnt;j++){
        int s=mj_name2id(sim.model,mjOBJ_JOINT,mj_id2name(g.model(),mjOBJ_JOINT,j));
        out<<','<<sim.data->qvel[sim.model->jnt_dofadr[s]];
    }
    for(double v:floorForce)out<<','<<v;
    for(double v:floorBrake)out<<','<<v;
    for(double v:toeTarget)out<<','<<v;
}

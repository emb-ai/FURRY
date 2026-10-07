#pragma once
#include "simulation.h"
#include "gmr.h"
#include <ostream>
#include <cmath>
#include <limits>

// Offline diagnostics only. Compact, consistently ordered robot state even
// when the physics model also contains fingers and free objects.
inline void AblationHeader(std::ostream& out) {
    out << ",segment,applying,contacts,foot_force,foot_slip,ref_available";
    for (auto field : {"q", "ref"}) for(int k=0;k<36;k++) out<<','<<field<<k;
    for(int k=0;k<35;k++) out<<",v"<<k;
}
inline void AblationRow(std::ostream& out, Simulation& sim, GmrRetargeter& g,
                        int segment, bool applying, bool referenceAvailable=true) {
    double forceSum=0, slipSum=0;
    for(int c=0;c<sim.data->ncon;c++) {
        const auto& contact=sim.data->contact[c];
        int a=sim.model->geom_bodyid[contact.geom1],b=sim.model->geom_bodyid[contact.geom2];
        int foot=a==0?b:(b==0?a:-1);
        if(foot<0)continue;
        const char* name=mj_id2name(sim.model,mjOBJ_BODY,foot);
        if(!name || (std::string(name)!="left_ankle_roll_link" && std::string(name)!="right_ankle_roll_link"))continue;
        double force[6],velocity[3];std::vector<double> jp(3*sim.model->nv);
        mj_contactForce(sim.model,sim.data,c,force);
        if(force[0]<=1)continue;
        mj_jac(sim.model,sim.data,jp.data(),nullptr,contact.pos,foot);
        mju_mulMatVec(velocity,jp.data(),sim.data->qvel,3,sim.model->nv);
        forceSum+=force[0];slipSum+=force[0]*std::hypot(velocity[0],velocity[1]);
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
}

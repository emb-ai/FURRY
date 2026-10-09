#pragma once
#include "gmr.h"
#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>

// Geometry of the floor-contact pads, expressed in each GMR toe task frame.
// The foot box is used for self-contact and must not set the floor baseline.
// No contacts from the moving physical robot are used to ground a reference.
class FootFloor {
    struct Pad {
        int side;
        int type;
        std::array<double,3> center, size;
        std::array<double,9> rotation;
    };
    std::vector<Pad> pads;
public:
    void Load(const mjModel* physical, const mjModel* reference) {
        pads.clear();
        int floor=mj_name2id(physical,mjOBJ_GEOM,"floor");
        if(floor<0 || physical->geom_type[floor]!=mjGEOM_PLANE)
            throw std::runtime_error("Foot grounding requires a named flat floor");
        if(physical->geom_bodyid[floor]!=0 || std::abs(physical->geom_pos[3*floor+2])>1e-12 ||
           std::abs(physical->geom_quat[4*floor+1])+std::abs(physical->geom_quat[4*floor+2])>1e-12)
            throw std::runtime_error("Foot grounding requires a horizontal floor at z=0");
        for(int side=0;side<2;side++) {
            std::string prefix=side==0?"left":"right";
            int ankle=mj_name2id(physical,mjOBJ_BODY,(prefix+"_ankle_roll_link").c_str());
            int toe=mj_name2id(reference,mjOBJ_BODY,(prefix+"_toe_link").c_str());
            int refAnkle=mj_name2id(reference,mjOBJ_BODY,(prefix+"_ankle_roll_link").c_str());
            if(ankle<0 || toe<0 || refAnkle<0 || reference->body_parentid[toe]!=refAnkle ||
               std::abs(reference->body_quat[4*toe]-1)>1e-12)
                throw std::runtime_error("Unexpected ankle/toe frame mapping");
            size_t before=pads.size();
            for(int g=0;g<physical->ngeom;g++) {
                if(physical->geom_bodyid[g]!=ankle)continue;
                if(!((physical->geom_contype[g]&physical->geom_conaffinity[floor]) ||
                     (physical->geom_contype[floor]&physical->geom_conaffinity[g])))continue;
                int type=physical->geom_type[g];
                if(type!=mjGEOM_CAPSULE && type!=mjGEOM_SPHERE && type!=mjGEOM_BOX)
                    throw std::runtime_error("Unsupported floor-contact foot geometry");
                Pad pad{};pad.side=side;pad.type=type;
                for(int a=0;a<3;a++) {
                    pad.center[a]=physical->geom_pos[3*g+a]-reference->body_pos[3*toe+a];
                    pad.size[a]=physical->geom_size[3*g+a];
                }
                mju_quat2Mat(pad.rotation.data(),physical->geom_quat+4*g);
                pads.push_back(pad);
            }
            if(pads.size()==before)throw std::runtime_error("Missing floor-contact foot pads");
        }
    }
    std::array<double,2> Heights(const TrackedPose& left,const TrackedPose& right)const {
        std::array<double,2> heights{std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()};
        const TrackedPose* feet[]={&left,&right};
        for(const auto& pad:pads) {
            const auto& foot=*feet[pad.side];
            double R[9],p[3],world[9];mju_quat2Mat(R,foot.quaternion.data());
            mju_rotVecQuat(p,pad.center.data(),foot.quaternion.data());
            mju_mulMatMat(world,R,pad.rotation.data(),3,3,3);
            double extent=pad.size[0];
            if(pad.type==mjGEOM_CAPSULE)extent+=pad.size[1]*std::abs(world[8]);
            if(pad.type==mjGEOM_BOX)extent=pad.size[0]*std::abs(world[6])+pad.size[1]*std::abs(world[7])+pad.size[2]*std::abs(world[8]);
            heights[pad.side]=std::min(heights[pad.side],foot.position[2]+p[2]-extent);
        }
        return heights;
    }
    std::array<double,2> Heights(GmrRetargeter& g)const {
        TrackedPose feet[2];
        for(int side=0;side<2;side++) {
            int body=g.tasks()[6+side].body;
            std::copy_n(g.data()->xpos+3*body,3,feet[side].position.begin());
            mju_mat2Quat(feet[side].quaternion.data(),g.data()->xmat+9*body);
        }
        return Heights(feet[0],feet[1]);
    }
};

// Apply after startup blending and camera servo. A blended joint pose can
// penetrate even when both endpoint poses were individually grounded.
// Only reference height changes; joint angles, XY/yaw commands and the
// physical simulation state are preserved. Scratch FK has no solver history.
class FootFloorGuard {
    FootFloor floor;
    const mjModel* model;
    std::unique_ptr<mjData,decltype(&mj_deleteData)> pose;
public:
    double lastLift=0;
    FootFloorGuard(const mjModel* physical,const mjModel* reference)
        :model(reference),pose(mj_makeData(reference),mj_deleteData) {
        if(!pose)throw std::runtime_error("Cannot allocate floor-reference FK");
        if(model->nq!=36)throw std::runtime_error("Expected G1 floor-reference model");
        floor.Load(physical,reference);
    }
    std::array<float,35> Correct(std::array<float,35> command) {
        for(float v:command)if(!std::isfinite(v))throw std::runtime_error("Invalid floor-reference command");
        auto* q=pose->qpos;q[0]=q[1]=0;q[2]=command[2];
        double roll[4]={std::cos(command[3]/2.),std::sin(command[3]/2.),0,0};
        double pitch[4]={std::cos(command[4]/2.),0,std::sin(command[4]/2.),0};
        mju_mulQuat(q+3,pitch,roll); // Floor clearance is independent of yaw.
        for(int k=0;k<29;k++)q[7+k]=command[6+k];
        mj_kinematics(model,pose.get());
        TrackedPose feet[2];
        for(int side=0;side<2;side++) {
            int body=mj_name2id(model,mjOBJ_BODY,side==0?"left_toe_link":"right_toe_link");
            std::copy_n(pose->xpos+3*body,3,feet[side].position.begin());
            mju_mat2Quat(feet[side].quaternion.data(),pose->xmat+9*body);
        }
        auto height=floor.Heights(feet[0],feet[1]);
        lastLift=std::max(0.,-std::min(height[0],height[1]));
        if(lastLift>0)command[2]=std::nextafter(float(double(command[2])+lastLift),std::numeric_limits<float>::infinity());
        return command;
    }
};

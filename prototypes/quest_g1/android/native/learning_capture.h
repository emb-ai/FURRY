#pragma once
#include <mujoco/mujoco.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include "policy_catalog.h"

// Observation snapshots precede the command's physics interval. Rendering is
// deferred to the desktop; the serialized model contains all compiled assets.
class LearningCapture {
    std::ofstream stream;
    std::unique_ptr<mjData,decltype(&mj_deleteData)> scratch{nullptr,mj_deleteData};
    std::filesystem::path folder;
    int camera=-1,segment=0,outcome=0;
    uint64_t rows=0;
    bool gazeValid=false;
    std::array<double,3> gaze{};
public:
    static bool InT(double x,double y){return (std::abs(x-.12)<=.09&&std::abs(y)<=.30)||(std::abs(x+.12)<=.15&&std::abs(y)<=.09);}
    static int Result(const mjModel* m,const mjData* d){
        if(mj_name2id(m,mjOBJ_NUMERIC,"collection_task")>=0&&m->nuserdata==8)return int(d->userdata[3]);
        int box=mj_name2id(m,mjOBJ_BODY,"t_box"),target=mj_name2id(m,mjOBJ_BODY,"t_target");
        if(box<0||target<0)return 0;
        int pelvis=mj_name2id(m,mjOBJ_BODY,"pelvis");
        if(!std::isfinite(d->qpos[2])||d->qpos[2]<.50||(pelvis>=0&&d->xmat[9*pelvis+8]<.70710678))return 2;
        int all=0,inside=0;
        for(double x=-.26;x<.21;x+=.02)for(double y=-.29;y<.30;y+=.02){
            if(!InT(x,y))continue;
            all++;
            double world[3],local[3],p[3]={x,y,0};
            mju_mulMatVec(world,d->xmat+9*target,p,3,3);mju_addTo3(world,d->xpos+3*target);mju_subFrom3(world,d->xpos+3*box);
            mju_mulMatTVec(local,d->xmat+9*box,world,3,3);inside+=InT(local[0],local[1]);
        }
        return all&&double(inside)/all>=.5?1:0;
    }
    void Start(const std::filesystem::path& path,const mjModel* m){
        folder=path;camera=mj_name2id(m,mjOBJ_CAMERA,"ego");segment=outcome=0;rows=0;gazeValid=false;
        if(camera<0)return;
        scratch.reset(mj_makeData(m));if(!scratch)throw std::runtime_error("Learning camera buffer failed");
        auto model=folder/"learning-model.mjb";mj_saveModel(m,model.c_str(),nullptr,0);
        if(!std::filesystem::exists(model)||std::filesystem::file_size(model)!=size_t(mj_sizeModel(m)))throw std::runtime_error("Learning model snapshot failed");
        std::ofstream metadata;metadata.exceptions(std::ios::badbit|std::ios::failbit);metadata.open(folder/"learning.json");
        metadata<<"{\"schema\":\"furry-visual-capture-v1\",\"mujoco_version\":\""<<mj_versionString()<<"\",\"model_sha256\":\""<<questpolicy::FileSha256(model.string())<<"\",\"camera\":\"ego\",\"camera_mode\":\"robot_head_pan_tilt_v1\",\"camera_fovy_degrees\":"<<m->cam_fovy[camera]<<",\"image_width\":640,\"image_height\":360,\"sample_timing\":\"pre_action\",\"action_schema\":\"body35_grip2_gaze2\",\"export_hz\":30}\n";
        metadata.close();stream.clear();stream.exceptions(std::ios::badbit|std::ios::failbit);stream.open(folder/"learning.csv");
        stream<<"frame_index,sequence,step,sim_time,segment,camera_valid,camera_x,camera_y,camera_z,camera_qw,camera_qx,camera_qy,camera_qz,gaze_yaw,gaze_pitch,outcome\n"<<std::setprecision(17);
    }
    void Gaze(const double* direction,bool valid){gazeValid=valid;if(valid)std::copy_n(direction,3,gaze.begin());}
    void Event(const std::string& name){
        if(name=="reset"||name=="calibrate"||name=="focus_pause"||name=="user_pause"||name=="menu_open"||name=="reference_space_change"||name=="body_skeleton_changed"||name=="observer_view"||name=="ego_world")segment++;
    }
    void Frame(const mjModel* m,const mjData* d,uint64_t sequence,int step){
        if(!stream.is_open())return;
        mju_copy(scratch->qpos,d->qpos,m->nq);mju_copy(scratch->mocap_pos,d->mocap_pos,3*m->nmocap);mju_copy(scratch->mocap_quat,d->mocap_quat,4*m->nmocap);mj_kinematics(m,scratch.get());mj_camlight(m,scratch.get());
        const double* base=scratch->cam_xmat+9*camera;double local[3]={0,0,-1};
        if(gazeValid)mju_mulMatTVec(local,base,gaze.data(),3,3);
        const double yaw=std::atan2(-local[0],-local[2]),pitch=std::atan2(local[1],std::hypot(local[0],local[2]));
        double by[4]={std::cos(yaw/2),0,std::sin(yaw/2),0},px[4]={std::cos(pitch/2),std::sin(pitch/2),0,0},qbase[4],qrel[4],q[4];
        mju_mat2Quat(qbase,base);mju_mulQuat(qrel,by,px);mju_mulQuat(q,qbase,qrel);
        int result=Result(m,d);if(!outcome&&result)outcome=result;
        stream<<rows++<<','<<sequence<<','<<step<<','<<d->time<<','<<segment<<','<<gazeValid;
        for(int k=0;k<3;k++)stream<<','<<scratch->cam_xpos[3*camera+k];
        for(double v:q)stream<<','<<v;
        stream<<','<<yaw<<','<<pitch<<','<<outcome<<'\n';
    }
    void FinalState(const mjModel*m,const mjData*d){if(!outcome)outcome=Result(m,d);}
    void Flush(){if(stream.is_open())stream.flush();}
    void Stop(){if(!stream.is_open())return;stream.close();std::ofstream f;f.exceptions(std::ios::badbit|std::ios::failbit);f.open(folder/"learning-result.json");f<<"{\"schema_version\":1,\"outcome\":"<<outcome<<",\"frames\":"<<rows<<"}\n";f.close();}
    void Abort()noexcept{stream.exceptions(std::ios::goodbit);if(stream.is_open())stream.close();}
};

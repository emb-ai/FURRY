#include "cup_placement_cue.h"
#include <iostream>
#include <stdexcept>

static void Check(bool ok,const char* reason){if(!ok)throw std::runtime_error(reason);}
int main(int argc,char** argv){
    try{
        Check(argc==2,"Expected scene XML");
        char error[2048]{};mjModel* m=mj_loadXML(argv[1],nullptr,error,sizeof(error));
        if(!m)throw std::runtime_error(error);
        mjData* d=mj_makeData(m);
        int cup=mj_name2id(m,mjOBJ_BODY,"cup"),joint=mj_name2id(m,mjOBJ_JOINT,"cup_free");
        int q=m->jnt_qposadr[joint],v=m->jnt_dofadr[joint];
        int goalId=mj_name2id(m,mjOBJ_NUMERIC,"cup_lab_goal");
        Check(goalId>=0&&m->numeric_size[goalId]==4,"Missing goal");
        auto* goal=m->numeric_data+m->numeric_adr[goalId];double gx=goal[0],gy=goal[1],gz=goal[2];
        auto pose=[&](double x,double y,double z){
            mj_resetData(m,d);d->qpos[q]=x;d->qpos[q+1]=y;d->qpos[q+2]=z;
            d->qpos[q+3]=1;d->qpos[q+4]=d->qpos[q+5]=d->qpos[q+6]=0;
            mj_forward(m,d);
        };
        pose(gx,gy,gz);Check(CupPlacementCue(m,d)[3]==1,"Supported upright cup");
        pose(gx,gy,gz+.10);Check(CupPlacementCue(m,d)[3]==0,"Held above table");
        pose(gx+.10,gy,gz);Check(CupPlacementCue(m,d)[3]==0,"Outside goal");
        pose(gx,gy,gz);d->qvel[v]=.1;mj_forward(m,d);
        Check(CupPlacementCue(m,d)[3]==0,"Moving cup");
        pose(gx,gy,gz);d->qvel[v+5]=.4;mj_forward(m,d);
        Check(CupPlacementCue(m,d)[3]==0,"Spinning cup");
        pose(gx,gy,gz);d->qpos[q+3]=std::cos(.3);d->qpos[q+4]=std::sin(.3);mj_forward(m,d);
        Check(CupPlacementCue(m,d)[3]==0,"Tipped cup");
        int marker=mj_name2id(m,mjOBJ_BODY,"cup_goal_marker");
        if(marker>=0){
            pose(gx,gy,gz);int mocap=m->body_mocapid[marker];
            d->mocap_pos[3*mocap+1]=.24;mj_forward(m,d);
            Check(CupPlacementCue(m,d)[3]==0,"Old circle location still scores");
            d->qpos[q+1]=.24;mj_forward(m,d);
            Check(CupPlacementCue(m,d)[3]==1,"Randomized circle location ignored");
        }
        pose(gx,gy,gz);int id=mj_name2id(m,mjOBJ_NUMERIC,"cup_lab_goal");m->numeric_size[id]=3;
        Check(CupPlacementCue(m,d)[3]==0,"Malformed optional cue");
        Check(m->body_mass[cup]>.134&&m->body_mass[cup]<.136,"Mug mass");
        mj_deleteData(d);mj_deleteModel(m);std::cout<<"Cup placement cue passed\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

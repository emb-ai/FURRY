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
        auto pose=[&](double x,double y,double z){
            mj_resetData(m,d);d->qpos[q]=x;d->qpos[q+1]=y;d->qpos[q+2]=z;
            d->qpos[q+3]=1;d->qpos[q+4]=d->qpos[q+5]=d->qpos[q+6]=0;
            mj_forward(m,d);
        };
        pose(.65,-.25,.75);Check(CupPlacementCue(m,d)[3]==1,"Supported upright cup");
        pose(.65,-.25,.85);Check(CupPlacementCue(m,d)[3]==0,"Held above table");
        pose(.75,-.25,.75);Check(CupPlacementCue(m,d)[3]==0,"Outside goal");
        pose(.65,-.25,.75);d->qvel[v]=.1;mj_forward(m,d);
        Check(CupPlacementCue(m,d)[3]==0,"Moving cup");
        pose(.65,-.25,.75);d->qvel[v+5]=.4;mj_forward(m,d);
        Check(CupPlacementCue(m,d)[3]==0,"Spinning cup");
        pose(.65,-.25,.75);d->qpos[q+3]=std::cos(.3);d->qpos[q+4]=std::sin(.3);mj_forward(m,d);
        Check(CupPlacementCue(m,d)[3]==0,"Tipped cup");
        pose(.65,-.25,.75);int id=mj_name2id(m,mjOBJ_NUMERIC,"cup_lab_goal");m->numeric_size[id]=3;
        Check(CupPlacementCue(m,d)[3]==0,"Malformed optional cue");
        Check(m->body_mass[cup]>.239&&m->body_mass[cup]<.241,"Mug mass");
        mj_deleteData(d);mj_deleteModel(m);std::cout<<"Cup placement cue passed\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

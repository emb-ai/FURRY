#include "rubik_task.h"
#include <iostream>
#include <vector>
static void Check(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
static void PoseLayer(mjModel* m,mjData* d,RubikTask& task,int face,double angle){
    d->qpos[task.qa[face]]=d->userdata[RubikTask::Start]+angle;
    double rotation[4];RubikTask::Rotation(rotation,face,angle);
    mj_forward(m,d);
    for(int i=0;i<8;i++){
        const double* state=d->userdata+RubikTask::Corners+i*7;
        double p[3],q[4];
        if(RubikTask::OnFace(state,face)){mju_rotVecQuat(p,state,rotation);mju_mulQuat(q,rotation,state+3);}
        else{mju_copy(p,state,3);mju_copy(q,state+3,4);}
        int j=m->body_jntadr[task.cubie[i]],qa=m->jnt_qposadr[j];
        for(int k=0;k<3;k++)d->qpos[qa+k]=d->xpos[3*task.core+k]+p[k];
        mju_copy(d->qpos+qa+3,q,4);
    }
    mj_forward(m,d);
}
static void Step(mjModel* m,mjData* d,RubikTask& task,int count){
    for(int k=0;k<count;k++){task.Begin(m,d,0,0);mj_step(m,d);task.Finish(m,d);}
}
int main(int argc,char** argv){
    try{
        Check(argc==2,"Expected isolated puzzle fixture");char error[2048]{};
        mjModel* m=mj_loadXML(argv[1],nullptr,error,sizeof(error));if(!m)throw std::runtime_error(error);
        mjData* d=mj_makeData(m);RubikTask task;task.Initialize(m);task.Reset(m,d);task.Activate(m,d,4);
        Check(!task.Solved(d),"Scramble reported solved");
        Check(RubikTask::MagneticTorque(.3,0)==0,"Magnet outside 15 degrees");
        Check(RubikTask::MagneticTorque(.2,0)<0&&RubikTask::MagneticTorque(-.2,0)>0,"Magnetic restoring direction");
        Check(std::abs(RubikTask::MagneticTorque(RubikTask::Pi/2+.2,0)-RubikTask::MagneticTorque(.2,0))<1e-12,"Quarter-turn magnetic periodicity");
        PoseLayer(m,d,task,4,12*RubikTask::Pi/180);Step(m,d,task,900);
        double snapDegrees=(d->qpos[task.qa[4]]-d->userdata[RubikTask::Start])*180/RubikTask::Pi;
        std::cout<<"Magnetic settling diagnostic: "<<snapDegrees<<"\n";
        Check(std::abs(snapDegrees)<1,"Magnetic settling from 12 degrees");
        mj_resetData(m,d);task.Reset(m,d);task.Activate(m,d,4);PoseLayer(m,d,task,4,25*RubikTask::Pi/180);Step(m,d,task,400);
        double freeDegrees=(d->qpos[task.qa[4]]-d->userdata[RubikTask::Start])*180/RubikTask::Pi;
        Check(std::abs(freeDegrees-25)<.5,"Free layer outside magnetic window");
        auto* snapshot=mj_copyData(nullptr,m,d);Step(m,d,task,400);std::vector<double> expected(d->qpos,d->qpos+m->nq);
        mj_copyData(d,m,snapshot);Step(m,d,task,400);
        for(int i=0;i<m->nq;i++)Check(std::abs(d->qpos[i]-expected[i])<1e-10,"Snapshot mechanism reconstruction");
        mj_deleteData(snapshot);
        mj_resetData(m,d);task.Reset(m,d);task.Activate(m,d,4);PoseLayer(m,d,task,4,RubikTask::Pi/2);
        std::vector<double> qpos(d->qpos,d->qpos+m->nq),qvel(d->qvel,d->qvel+m->nv);
        std::vector<double> binding(m->eq_data,m->eq_data+m->neq*mjNEQDATA);task.Finish(m,d);
        for(int i=0;i<m->nq;i++)Check(d->qpos[i]==qpos[i],"Turn commit teleported a body");
        for(int i=0;i<m->nv;i++)Check(d->qvel[i]==qvel[i],"Turn commit overwrote a velocity");
        for(int i=0;i<8;i++)if(d->eq_active[task.layer[4][i]])for(int k=3;k<10;k++){
            int address=task.layer[4][i]*mjNEQDATA+k;
            Check(std::abs(m->eq_data[address]-binding[address])<1e-12,"Turn commit changed physical binding");
        }
        Check(d->userdata[RubikTask::Active]==4,"Quarter turn locked the hinge");
        // External load fixture: apply body torques to actual corner bodies.
        // No hinge motor or qpos/velocity writes after reset.
        mj_resetData(m,d);task.Reset(m,d);task.Activate(m,d,4);
        for(int step=0;step<1400;step++){
            mju_zero(d->xfrc_applied,6*m->nbody);
            double goal=80*RubikTask::Pi/180*std::min(1.,step/500.);
            double torque=step<800?std::clamp(.015*(goal-d->qpos[task.qa[4]])-.0015*d->qvel[task.va[4]],-.012,.012):0;
            for(int i=0;i<8;i++)if(RubikTask::OnFace(d->userdata+RubikTask::Corners+i*7,4))d->xfrc_applied[6*task.cubie[i]+5]=torque/4;
            Step(m,d,task,1);
        }
        double loadedDegrees=d->qpos[task.qa[4]]*180/RubikTask::Pi;
        Check(std::abs(loadedDegrees-90)<1,"Real corner bodies failed externally loaded quarter-turn");
        double turn[4];RubikTask::Rotation(turn,4,RubikTask::Pi/2);
        for(int i=0;i<8;i++){
            const double* state=m->numeric_data+m->numeric_adr[task.initial]+i*7;
            double expected[4],inverse[4],errorQuat[4],errorVel[3];
            if(RubikTask::OnFace(state,4))mju_mulQuat(expected,turn,state+3);else mju_copy(expected,state+3,4);
            mju_negQuat(inverse,expected);int address=m->jnt_qposadr[m->body_jntadr[task.cubie[i]]];
            mju_mulQuat(errorQuat,d->qpos+address+3,inverse);mju_quat2Vel(errorVel,errorQuat,1);
            Check(mju_norm3(errorVel)<RubikTask::Pi/180,"Corner color body did not physically rotate");
        }
        std::cout<<"external_body_load_quarter_deg="<<loadedDegrees<<" (fixed core fixture, no hand proof)\n";
        mj_resetData(m,d);task.Reset(m,d);task.Activate(m,d,4);PoseLayer(m,d,task,4,0);
        for(auto move:std::array<std::array<int,2>,4>{{{0,1},{3,-1},{4,1},{0,-1}}}){
            Check(task.Activate(m,d,move[0]),"Aligned layer switch");PoseLayer(m,d,task,move[0],move[1]*RubikTask::Pi/2);task.Finish(m,d);
        }
        Check(task.Solved(d),"Inverse scramble did not solve all six colors");
        for(int i=0;i<m->neq;i++)Check(m->eq_obj1id[i]==task.core||
            std::string(mj_id2name(m,mjOBJ_BODY,m->eq_obj1id[i])).rfind("rubik_layer_",0)==0,"External grasp weld");
        std::cout<<"magnetic_12_deg_final="<<snapDegrees<<" free_25_deg_final="<<freeDegrees
                 <<" snapshots_and_no_teleport_and_inverse_scramble_passed\n";
        mj_deleteData(d);mj_deleteModel(m);
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

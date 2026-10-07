// GMR (MIT, Yanjie Ze 2025), Mink (Apache-2.0) native adaptation.
// Python/Mink dependencies are replaced by MuJoCo math and its box-QP solver.
// The box constraints are equivalent to Mink ConfigurationLimit for this model
// (one free root and 29 hinge joints). No analytic gait or independent arm PD.
#include "gmr.h"
#include <fstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
namespace {
void Skew(double* out,const double* v){double a[9]={0,-v[2],v[1],v[2],0,-v[0],-v[1],v[0],0};std::copy_n(a,9,out);}
void Mul(double* c,const double*a,const double*b){mju_mulMatMat(c,a,b,3,3,3);}
void SOInverse(double* inv,const double* w){
    double W[9],W2[9];Skew(W,w);Mul(W2,W,W);double t2=mju_dot3(w,w);
    double coeff=t2<1e-10?1./12:(1-.5*std::sqrt(t2)/std::tan(.5*std::sqrt(t2)))/t2;
    for(int i=0;i<9;i++)inv[i]=(i%4==0?1.:0)-.5*W[i]+coeff*W2[i];
}
void Residual(double* e,const double* pos,const double* rot,const TrackedPose& target){
    double tr[9],rel[9],q[4],p[3],t[3],inv[9];mju_quat2Mat(tr,target.quaternion.data());
    mju_mulMatTMat(rel,rot,tr,3,3,3);mju_mat2Quat(q,rel);mju_quat2Vel(e+3,q,1.);
    mju_sub3(p,target.position.data(),pos);mju_mulMatTVec(t,rot,p,3,3);SOInverse(inv,e+3);mju_mulMatVec(e,inv,t,3,3);
}
void LogJacobian(double* out,const double* e){
    std::fill_n(out,36,0.);double t2=mju_dot3(e+3,e+3);
    // Match Mink's small-angle branch exactly, including translation block.
    if(t2<1e-10){for(int i=0;i<6;i++)out[i*6+i]=1;return;}
    double t=std::sqrt(t2),B=(t-std::sin(t))/(t2*t),C=(1-.5*t2-std::cos(t))/(t2*t2),D=(2*t-3*std::sin(t)+t*std::cos(t))/(2*t2*t2*t);
    double V[9],W[9],VW[9],WV[9],WVW[9],VWW[9],a[9],b[9],Q[9],inv[9],tmp[9],cross[9];
    Skew(V,e);Skew(W,e+3);Mul(VW,V,W);mju_transpose(WV,VW,3,3);Mul(WVW,WV,W);Mul(VWW,VW,W);Mul(a,WVW,W);Mul(b,W,WVW);
    for(int r=0;r<3;r++)for(int c=0;c<3;c++){int i=3*r+c;Q[i]=.5*V[i]+B*(WV[i]+VW[i]+WVW[i])-C*(VWW[i]-VWW[3*c+r]-3*WVW[i])+D*(a[i]+b[i]);}
    SOInverse(inv,e+3);Mul(tmp,inv,Q);Mul(cross,tmp,inv);
    for(int r=0;r<3;r++)for(int c=0;c<3;c++){out[r*6+c]=out[(r+3)*6+c+3]=inv[3*r+c];out[r*6+c+3]=-cross[3*r+c];}
}
}
GmrRetargeter::GmrRetargeter(const std::string& assets){
    char error[1024];model_=mj_loadXML((assets+"/gmr_model.xml").c_str(),nullptr,error,sizeof(error));
    if(!model_)throw std::runtime_error(error);
    data_=mj_makeData(model_);
    if(model_->nq!=36 || model_->nv!=35)throw std::runtime_error("Expected GMR G1 29-DoF free-base model");
    std::ifstream f(assets+"/gmr_config.txt");std::string magic;int count;f>>magic>>count>>assumedHeight>>ground;
    if(!f || magic!="GMR1" || count!=14)throw std::runtime_error("Invalid GMR config");
    for(int i=0;i<count;i++){
        GmrTask task;f>>task.human>>task.robot>>task.scale;for(auto&v:task.costs)f>>v;for(auto&v:task.offset.position)f>>v;for(auto&v:task.offset.quaternion)f>>v;
        task.body=mj_name2id(model_,mjOBJ_BODY,task.robot.c_str());if(!f || task.body<0)throw std::runtime_error("Invalid GMR task");tasks_.push_back(task);
    }
    cameraBody=mj_name2id(model_,mjOBJ_BODY,"torso_link");
    if(cameraBody<0)throw std::runtime_error("Missing camera body");
    cameraLocal.position={.08,0,.43};
    double cameraRotation[9]={0,0,-1,-1,0,0,0,1,0};
    mju_mat2Quat(cameraLocal.quaternion.data(),cameraRotation);
    targets.resize(count);Reset();
}
GmrRetargeter::~GmrRetargeter(){mj_deleteData(data_);mj_deleteModel(model_);}
void GmrRetargeter::Reset(const double* qpos){cameraEnabled=false;mj_resetData(model_,data_);if(qpos)std::copy_n(qpos,model_->nq,data_->qpos);mj_kinematics(model_,data_);mj_comPos(model_,data_);}
void GmrRetargeter::SetTargets(const std::vector<TrackedPose>& value){
    if(value.size()!=tasks_.size())throw std::runtime_error("Incomplete GMR skeleton");
    for(const auto&p:value){for(double v:p.position)if(!std::isfinite(v))throw std::runtime_error("Invalid body position");for(double v:p.quaternion)if(!std::isfinite(v))throw std::runtime_error("Invalid body rotation");}
    targets=value;
}
void GmrRetargeter::SetHumanTargets(const std::vector<TrackedPose>& human,double height,bool offsetToGround){
    if(human.size()!=tasks_.size() || !(height>0))throw std::runtime_error("Invalid human skeleton");
    // Exact upstream height rule. Device normalization is a separate adapter.
    const double ratio=height/assumedHeight;auto out=human;
    for(size_t i=0;i<out.size();i++){
        const auto&t=tasks_[i];double offset[3]={t.offset.position[0],t.offset.position[1],t.offset.position[2]-ground},translated[3];
        mju_mulQuat(out[i].quaternion.data(),human[i].quaternion.data(),t.offset.quaternion.data());
        mju_rotVecQuat(translated,offset,out[i].quaternion.data());
        for(int a=0;a<3;a++)out[i].position[a]=human[0].position[a]*tasks_[0].scale*ratio+(human[i].position[a]-human[0].position[a])*t.scale*ratio+translated[a];
    }
    if(offsetToGround){double floor=1e10;for(size_t i=0;i<out.size();i++)if(tasks_[i].human.find("Foot")!=std::string::npos)floor=std::min(floor,out[i].position[2]);for(auto&p:out)p.position[2]+=.1-floor;}
    SetTargets(out);
}
TrackedPose GmrRetargeter::CameraPose()const{
    TrackedPose pose;double offset[3],rotation[4];
    mju_mat2Quat(rotation,data_->xmat+9*cameraBody);
    mju_rotVecQuat(offset,cameraLocal.position.data(),rotation);
    for(int i=0;i<3;i++)pose.position[i]=data_->xpos[3*cameraBody+i]+offset[i];
    mju_mulQuat(pose.quaternion.data(),rotation,cameraLocal.quaternion.data());
    return pose;
}
std::array<double,3> GmrRetargeter::AlignTargetsToCameraXY(const TrackedPose& target){
    auto referenceCamera=CameraPose();
    std::array<double,3> shift{target.position[0]-referenceCamera.position[0],
                               target.position[1]-referenceCamera.position[1],0};
    for(double v:shift)if(!std::isfinite(v))throw std::runtime_error("Invalid camera target translation");
    // Both task targets and their warm start undergo the same rigid shift.
    // This moves only the kinematic reference, never the physical robot.
    for(auto& p:targets)for(int a=0;a<2;a++)p.position[a]+=shift[a];
    for(int a=0;a<2;a++)data_->qpos[a]+=shift[a];
    mj_kinematics(model_,data_);mj_comPos(model_,data_);
    return shift;
}
void GmrRetargeter::SetCameraTarget(const TrackedPose& target){
    for(double v:target.position)if(!std::isfinite(v))throw std::runtime_error("Invalid camera position");
    for(double v:target.quaternion)if(!std::isfinite(v))throw std::runtime_error("Invalid camera orientation");
    cameraTarget=target;mju_normalize4(cameraTarget.quaternion.data());cameraEnabled=true;
}
double GmrRetargeter::Error(int stage){
    double sum=0;for(size_t i=0;i<tasks_.size();i++){const auto&t=tasks_[i];if(!t.costs[2*stage]&&!t.costs[2*stage+1])continue;double e[6];Residual(e,data_->xpos+3*t.body,data_->xmat+9*t.body,targets[i]);for(double v:e)sum+=v*v;}if(stage==1 && cameraEnabled){
        auto pose=CameraPose();double rotation[9],e[6];mju_quat2Mat(rotation,pose.quaternion.data());
        Residual(e,pose.position.data(),rotation,cameraTarget);for(double v:e)sum+=v*v;
    }
    return std::sqrt(sum);
}
void GmrRetargeter::Step(int stage){
    constexpr int n=35;double H[n*n]={},g[n]={},dq[n]={},factor[n*(n+7)]={},lower[n],upper[n];int index[n];
    double jp[3*n],jr[3*n],bodyJ[6*n],J[6*n],jlog[36],e[6];double damping=.5;
    for(size_t i=0;i<tasks_.size();i++){
        const auto&t=tasks_[i];double pc=t.costs[2*stage],rc=t.costs[2*stage+1];if(!pc&&!rc)continue;
        const double* R=data_->xmat+9*t.body;Residual(e,data_->xpos+3*t.body,R,targets[i]);LogJacobian(jlog,e);
        mj_jacBody(model_,data_,jp,jr,t.body);
        mju_mulMatTMat(bodyJ,R,jp,3,3,n);mju_mulMatTMat(bodyJ+3*n,R,jr,3,3,n);
        mju_mulMatMat(J,jlog,bodyJ,6,6,n);
        for(int r=0;r<6;r++){
            double weight=r<3?pc:rc,ew=weight*e[r];damping+=ew*ew;
            for(int a=0;a<n;a++){
                double ja=-J[r*n+a]*weight;g[a]+=ja*ew;
                for(int b=0;b<n;b++)H[a*n+b]+=ja*(-J[r*n+b]*weight);
            }
        }
    }
    if(stage==1 && cameraEnabled){
        auto pose=CameraPose();double R[9];mju_quat2Mat(R,pose.quaternion.data());
        Residual(e,pose.position.data(),R,cameraTarget);LogJacobian(jlog,e);
        mj_jac(model_,data_,jp,jr,pose.position.data(),cameraBody);
        mju_mulMatTMat(bodyJ,R,jp,3,3,n);mju_mulMatTMat(bodyJ+3*n,R,jr,3,3,n);
        mju_mulMatMat(J,jlog,bodyJ,6,6,n);
        for(int r=0;r<6;r++){
            double weight=r<3?cameraPositionCost:cameraRotationCost,ew=weight*e[r];damping+=ew*ew;
            for(int a=0;a<n;a++){
                double ja=-J[r*n+a]*weight;g[a]+=ja*ew;
                for(int b=0;b<n;b++)H[a*n+b]+=ja*(-J[r*n+b]*weight);
            }
        }
    }
    for(int i=0;i<n;i++){H[i*n+i]+=damping;lower[i]=-mjMAXVAL;upper[i]=mjMAXVAL;}
    for(int j=0;j<model_->njnt;j++)if(model_->jnt_type[j]==mjJNT_HINGE && model_->jnt_limited[j]){
        int v=model_->jnt_dofadr[j],q=model_->jnt_qposadr[j];lower[v]=.95*(model_->jnt_range[2*j]-data_->qpos[q]);upper[v]=.95*(model_->jnt_range[2*j+1]-data_->qpos[q]);
    }
    if(mju_boxQP(dq,factor,index,H,g,n,lower,upper)<0)throw std::runtime_error("GMR QP failed");
    mj_integratePos(model_,data_->qpos,dq,1.);mj_kinematics(model_,data_);mj_comPos(model_,data_);iterations++;
}
void GmrRetargeter::Solve(){
    iterations=0;
    for(int stage=0;stage<2;stage++){
        double before=Error(stage);Step(stage);double after=Error(stage);int repeat=0;
        while(before-after>.001 && repeat<10){before=after;Step(stage);after=Error(stage);repeat++;}
    }
    error=Error(1);
}

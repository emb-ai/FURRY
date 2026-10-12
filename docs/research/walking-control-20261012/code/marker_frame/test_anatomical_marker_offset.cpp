#include "anatomical_marker_offset.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

using Vec=AnatomicalMarkerOffset::Vec;
using Quat=AnatomicalMarkerOffset::Quat;
static Vec Add(Vec a,const Vec& b){for(int k=0;k<3;k++)a[k]+=b[k];return a;}
static Vec Sub(Vec a,const Vec& b){for(int k=0;k<3;k++)a[k]-=b[k];return a;}
static Vec Scale(Vec a,double s){for(auto& x:a)x*=s;return a;}
static Vec Rotate(const Quat& q,const Vec& p){Vec r;mju_rotVecQuat(r.data(),p.data(),q.data());return r;}
static Quat Mul(const Quat& a,const Quat& b){Quat r;mju_mulQuat(r.data(),a.data(),b.data());return r;}
static Quat Axis(int axis,double angle){Quat q{std::cos(angle/2),0,0,0};q[axis+1]=std::sin(angle/2);return q;}
static Vec Position(const mjData* d,int b){return {d->xpos[3*b],d->xpos[3*b+1],d->xpos[3*b+2]};}
static Quat Rotation(const mjData* d,int b){Quat q;mju_mat2Quat(q.data(),d->xmat+9*b);return q;}
static double Error(const Vec& a,const Vec& b){Vec d=Sub(a,b);return mju_norm3(d.data());}
static std::map<std::string,double> errors;
static void Check(const std::string& name,const Vec& actual,const Vec& expected,double tol=2e-12){
    double error=Error(actual,expected);errors[name]=std::max(errors[name],error);
    if(error>tol)throw std::runtime_error(name+": "+std::to_string(error));
}
static bool RejectsMovingMarker(mjModel* m,mjData* d,int root,int ankle){
    try {AnatomicalMarkerOffset f;f.CalibrateFromModel(m,d,root,root,ankle,{0,0,0});}
    catch(const std::runtime_error&){return true;}return false;
}
int main(int argc,char**argv){try{
    if(argc!=3)throw std::runtime_error("Usage: test model.xml result.json");
    char err[2048];auto* m=mj_loadXML(argv[1],nullptr,err,sizeof(err));
    if(!m)throw std::runtime_error(err);auto* d=mj_makeData(m);mj_resetData(m,d);
    d->qpos[2]=.8;mj_kinematics(m,d);int root=mj_name2id(m,mjOBJ_BODY,"pelvis");
    double legacySpuriousAnkleMotion=0,maxRootTiltDeltaZ=0;int cases=0;bool rejected=true;
    std::array<Vec,2> rigidFromModel{},morphology{};
    for(int side=0;side<2;side++) {
        std::string prefix=side?"right_":"left_";
        int ankle=mj_name2id(m,mjOBJ_BODY,(prefix+"ankle_roll_link").c_str());
        int marker=mj_name2id(m,mjOBJ_BODY,(prefix+"toe_link").c_str());
        // Anatomical mismatch includes all three coordinates. It deliberately
        // differs per side; no symmetric-foot or horizontal-foot assumption.
        Vec mapped=Add(Scale(Sub(Position(d,ankle),Position(d,root)),1.07),
                       {side?.021:-.031,side?-.046:.038,.013});
        AnatomicalMarkerOffset f;f.CalibrateFromModel(m,d,root,ankle,marker,mapped);
        auto p=Position(d,root),a=Position(d,ankle),t=Position(d,marker);
        auto Q=Rotation(d,root),R=Rotation(d,marker);
        rigidFromModel[side]=f.RigidInMarker();morphology[side]=f.MorphologyInRoot();
        Check("neutral_target",f.Target(p,Q,mapped,R),t);cases++;
        Check("FK_rigid_vector",Rotate(R,f.RigidInMarker()),Sub(t,a));cases++;
        rejected &= RejectsMovingMarker(m,d,root,ankle);
        Quat inv; mju_negQuat(inv.data(),R.data());Vec legacy=Rotate(inv,Sub(Sub(t,p),mapped));
        for(int axis=0;axis<3;axis++)for(double angle:{-.8,-.2,0.,.2,.8}) {
            auto foot=Mul(Axis(axis,angle),R);auto changed=f.Target(p,Q,mapped,foot);
            // Changing foot orientation must move only its physical rigid lever.
            Check("foot_rotation_true_lever",Sub(changed,t),Sub(Rotate(foot,f.RigidInMarker()),Rotate(R,f.RigidInMarker())));
            Check("foot_rotation_ankle_fixed",Sub(changed,Rotate(foot,f.RigidInMarker())),a);cases+=2;
            auto old=Add(Add(p,mapped),Rotate(foot,legacy));
            legacySpuriousAnkleMotion=std::max(legacySpuriousAnkleMotion,Error(Sub(old,Rotate(foot,f.RigidInMarker())),a));
        }
        for(int axis=0;axis<3;axis++)for(double angle:{-.6,-.15,.15,.6}) {
            auto tilted=Mul(Axis(axis,angle),Q);auto changed=f.Target(p,tilted,mapped,R);
            auto expected=Add(Add(Add(p,mapped),Rotate(tilted,f.MorphologyInRoot())),Rotate(R,f.RigidInMarker()));
            Check("root_tilt_full_vector",changed,expected);
            Check("root_tilt_rigid_unchanged",Sub(changed,Add(Add(p,mapped),Rotate(tilted,f.MorphologyInRoot()))),Rotate(R,f.RigidInMarker()));cases+=2;
            maxRootTiltDeltaZ=std::max(maxRootTiltDeltaZ,std::abs(changed[2]-t[2]));
        }
        for(int n=0;n<27;n++) {
            // A general 3D scene transform, not just heading/yaw.
            auto world=Mul(Mul(Axis(2,.17*n-.9),Axis(1,.13*n-.7)),Axis(0,.11*n-.5));
            Vec translation{.1*n-1,.3-.07*n,-.2+.09*n};
            auto p2=Add(Rotate(world,p),translation),a2=Add(Rotate(world,a),translation),t2=Add(Rotate(world,t),translation);
            auto Q2=Mul(world,Q),R2=Mul(world,R);auto mapped2=Rotate(world,mapped);
            Check("global_SE3_target",f.Target(p2,Q2,mapped2,R2),t2);
            AnatomicalMarkerOffset g;g.Calibrate(p2,Q2,a2,t2,R2,mapped2);
            Check("global_SE3_morphology",g.MorphologyInRoot(),f.MorphologyInRoot());
            Check("global_SE3_rigid",g.RigidInMarker(),f.RigidInMarker());
            Check("transformed_bind_neutral",g.Target(p2,Q2,mapped2,R2),t2);cases+=4;
        }
    }
    // Independent generic bind with nonidentity task and pelvis frames proves
    // that rho cannot be replaced with an unrotated model-local body_pos.
    {
        Vec p{.3,-.2,.9},a{-.1,.05,.07},mapped{-.38,.24,-.79};
        auto Q=Mul(Axis(1,.2),Axis(2,-.4)),R=Mul(Axis(0,.5),Axis(1,-.7));
        Vec physicalMarker{.072,-.034,.018},t=Add(a,Rotate(R,physicalMarker));
        AnatomicalMarkerOffset f;f.Calibrate(p,Q,a,t,R,mapped);
        Check("nonidentity_marker_rigid",f.RigidInMarker(),physicalMarker);
        Check("nonidentity_bind_neutral",f.Target(p,Q,mapped,R),t);cases+=2;
    }
    if(!rejected||legacySpuriousAnkleMotion<.01)throw std::runtime_error("Test does not detect the legacy defect or moving-marker attachment");
    std::ofstream out(argv[2]);out<<std::setprecision(17)<<"{\n  \"status\":\"PASS\",\n  \"numeric_cases\":"<<cases<<",\n  \"moving_marker_attachment_rejected\":true,\n  \"max_error_m\":{";
    bool first=true;for(const auto& kv:errors){if(!first)out<<',';out<<"\n    \""<<kv.first<<"\":"<<kv.second;first=false;}out<<"\n  },\n  \"synthetic_legacy_spurious_ankle_motion_m\":"<<legacySpuriousAnkleMotion<<",\n  \"synthetic_root_tilt_max_vertical_change_m\":"<<maxRootTiltDeltaZ<<",\n  \"model_derived_rigid_vectors\":[";
    for(int s=0;s<2;s++){if(s)out<<',';out<<'[';for(int k=0;k<3;k++){if(k)out<<',';out<<rigidFromModel[s][k];}out<<']';}out<<"],\n  \"scope\":\"Target-frame invariants only. Before floor/IK. Root morphology remains an approximate chain-length correction. No gait or contact feasibility claim.\"\n}\n";
    mj_deleteData(d);mj_deleteModel(m);std::cout<<"PASS "<<cases<<" numerical assertions and moving-marker rejection\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

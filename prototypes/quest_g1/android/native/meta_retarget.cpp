#include "meta_retarget.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
namespace {
double Distance(const TrackedPose&a,const TrackedPose&b){double d[3];mju_sub3(d,a.position.data(),b.position.data());return mju_norm3(d);}
void Normalize(double* v){if(mju_normalize3(v)<1e-6)throw std::runtime_error("Degenerate body skeleton");}
}
void MetaRetargeter::Calibrate(const mjModel* model,const mjData* data,const TrackingFrame& input){
    calibrated=false;
    if(!input.body.valid)throw std::runtime_error("Full body tracking required");
    auto* gm=gmr.model();auto* gd=gmr.data();gmr.Reset();gmr.ClearCameraTarget();gd->qpos[2]=.8;
    for(auto side:{"left","right"}){
        bool left=std::string(side)=="left";
        int roll=mj_name2id(gm,mjOBJ_JOINT,(std::string(side)+"_shoulder_roll_joint").c_str());
        int elbow=mj_name2id(gm,mjOBJ_JOINT,(std::string(side)+"_elbow_joint").c_str());
        gd->qpos[gm->jnt_qposadr[roll]]=(left?1:-1)*1.5707963267948966;
        gd->qpos[gm->jnt_qposadr[elbow]]=1.5707963267948966;
    }
    mj_kinematics(gm,gd);mj_comPos(gm,gd);
    std::array<TrackedPose,14> robot;
    for(int i=0;i<14;i++){int b=gmr.tasks()[i].body;std::copy_n(gd->xpos+3*b,3,robot[i].position.begin());mju_mat2Quat(robot[i].quaternion.data(),gd->xmat+9*b);}
    const auto& rest=input.body.rest;
    // Meta's T-pose is Y-up. Spine curvature is anatomy, not a reference-space
    // tilt: deriving up from pelvis->chest tilts every limb's bind correction.
    // Use the horizontal shoulder axis only for the neutral heading.
    double left[3],up[3]={0,1,0},forward[3],restBasis[9];
    mju_sub3(left,rest[8].position.data(),rest[9].position.data());left[1]=0;Normalize(left);
    mju_cross(forward,left,up);Normalize(forward);
    std::copy_n(forward,3,restBasis);std::copy_n(left,3,restBasis+3);std::copy_n(up,3,restBasis+6);
    double headRotation[9];mju_quat2Mat(headRotation,input.head.quaternion.data());
    double fx=-headRotation[2],fz=-headRotation[8],length=std::hypot(fx,fz);if(length<.2)throw std::runtime_error("Look forward to calibrate");fx/=length;fz/=length;
    double heading[9]={fx,0,fz,fz,0,-fx,0,1,0};
    double w=data->qpos[3],x=data->qpos[4],y=data->qpos[5],z=data->qpos[6];
    double yaw=std::atan2(2*(w*z+x*y),1-2*(y*y+z*z));double yawQ[4]={std::cos(yaw/2),0,0,std::sin(yaw/2)},yawR[9];mju_quat2Mat(yawR,yawQ);
    mju_mulMatMat(basis,yawR,heading,3,3,3);mju_mat2Quat(basisQuat,basis);
    double humanLeg=(Distance(rest[2],rest[4])+Distance(rest[4],rest[6])+Distance(rest[3],rest[5])+Distance(rest[5],rest[7]))*.5;
    double humanArm=(Distance(rest[8],rest[10])+Distance(rest[10],rest[12])+Distance(rest[9],rest[11])+Distance(rest[11],rest[13]))*.5;
    if(humanLeg<.3 || humanLeg>1.5 || humanArm<.25 || humanArm>1.1)throw std::runtime_error("Invalid body scale; wait for body calibration");
    double restFoot=.5*(rest[6].position[1]+rest[7].position[1]);
    double humanPelvisHeight=rest[0].position[1]-restFoot;
    // Meta samples FOOT_ANKLE; GMR foot tasks are toe frames 2 cm lower.
    // Scale matching anatomical landmarks, then apply ankle->toe offsets below.
    int leftAnkle=mj_name2id(gm,mjOBJ_BODY,"left_ankle_roll_link"),rightAnkle=mj_name2id(gm,mjOBJ_BODY,"right_ankle_roll_link");
    double ankleHeight=.5*(gd->xpos[3*leftAnkle+2]+gd->xpos[3*rightAnkle+2]);
    double robotPelvisHeight=robot[0].position[2]-ankleHeight;
    if(humanPelvisHeight<.3)throw std::runtime_error("Invalid skeleton up axis/height");
    rootScale=robotPelvisHeight/humanPelvisHeight;
    // The GMR shoulder-yaw task frame lies below the anatomical shoulder pivot.
    // Use the pitch pivot for limb length, retain local offsets for task frames.
    double robotReach=0;
    for(int h=0;h<2;h++){
        int b=mj_name2id(gm,mjOBJ_BODY,h==0?"left_shoulder_pitch_link":"right_shoulder_pitch_link");
        double upper[3];mju_sub3(upper,gd->xpos+3*b,robot[10+h].position.data());
        robotReach+=(mju_norm3(upper)+Distance(robot[10+h],robot[12+h]))*.5;
    }
    double armScale=robotReach/humanArm;
    footHeight=ankleHeight;
    for(int i=0;i<14;i++){
        scales[i]=i>=8?armScale:rootScale;
        double restR[9],aligned[9],robotR[9],offsetR[9];mju_quat2Mat(restR,rest[i].quaternion.data());mju_mulMatMat(aligned,restBasis,restR,3,3,3);mju_quat2Mat(robotR,robot[i].quaternion.data());
        mju_mulMatTMat(offsetR,aligned,robotR,3,3,3);mju_mat2Quat(offsets[i].quaternion.data(),offsetR);
        double rel[3],mapped[3],difference[3];mju_sub3(rel,rest[i].position.data(),rest[0].position.data());mju_mulMatVec(mapped,restBasis,rel,3,3);
        for(int a=0;a<3;a++)difference[a]=robot[i].position[a]-robot[0].position[a]-scales[i]*mapped[a];
        mju_mulMatTVec(offsets[i].position.data(),robotR,difference,3,3);
    }
    // Warm-start the genuine GMR model from the achieved physical robot pose.
    std::array<double,36> initial{};std::copy_n(data->qpos,7,initial.begin());
    for(int j=1;j<gm->njnt;j++){
        const char* name=mj_id2name(gm,mjOBJ_JOINT,j);int source=mj_name2id(model,mjOBJ_JOINT,name);
        if(source<0)throw std::runtime_error("GMR/physics joint mapping mismatch");initial[gm->jnt_qposadr[j]]=data->qpos[model->jnt_qposadr[source]];
    }
    gmr.Reset(initial.data());translationOrigin=input.body.joints[0].position;rootOrigin={data->qpos[0],data->qpos[1],data->qpos[2]};
    cameraOrigin=gmr.CameraPose();cameraHeadOrigin=input.head;
    double eyeHeight=input.head.position[1];
    if(!std::isfinite(eyeHeight) || eyeHeight<.7 || eyeHeight>2.4)
        throw std::runtime_error("Invalid standing eye height: calibrate upright");
    cameraScale=cameraOrigin.position[2]/eyeHeight;
    double alignedHead[4],inverse[4];mju_mulQuat(alignedHead,basisQuat,input.head.quaternion.data());
    mju_negQuat(inverse,alignedHead);mju_mulQuat(cameraRotationOffset.data(),inverse,cameraOrigin.quaternion.data());
    skeletonVersion=input.body.skeleton_version;lastTime=0;calibrated=true;
}
const std::array<float,35>& MetaRetargeter::Solve(const TrackingFrame& input){
    if(!calibrated || !input.body.valid)throw std::runtime_error("No calibrated full body sample");
    if(input.body.skeleton_version!=skeletonVersion){calibrated=false;throw std::runtime_error("Body proportions changed: press A to recalibrate");}
    if(lastTime && input.body.time_ns<=lastTime)return mimic;
    std::vector<TrackedPose> targets(14);double pelvisDelta[3],translation[3];
    // Whole-body translation belongs to the pelvis. Adding HMD displacement
    // here also translates planted feet when the head moves relative to the
    // torso, and mixes predicted head time with the returned body timestamp.
    mju_sub3(pelvisDelta,input.body.joints[0].position.data(),translationOrigin.data());
    mju_mulMatVec(translation,basis,pelvisDelta,3,3);
    // Whole-world travel uses the same similarity scale as the stationary
    // first-person world. Anatomical limb/height scales remain separate.
    double travelScale=cameraTracking?cameraScale:rootScale;
    double root[3]={rootOrigin[0]+travelScale*translation[0],rootOrigin[1]+travelScale*translation[1],0};
    // Grounded GMR mode: preserve pelvis height above the lower foot. Estimated
    // feet are used explicitly; there is no hidden generated gait in this app.
    double floor=std::min(input.body.joints[6].position[1],input.body.joints[7].position[1]);
    root[2]=footHeight+rootScale*(input.body.joints[0].position[1]-floor);
    for(int i=0;i<14;i++){
        double alignedQ[4],rel[3],mapped[3],offset[3];mju_mulQuat(alignedQ,basisQuat,input.body.joints[i].quaternion.data());mju_mulQuat(targets[i].quaternion.data(),alignedQ,offsets[i].quaternion.data());
        mju_sub3(rel,input.body.joints[i].position.data(),input.body.joints[0].position.data());mju_mulMatVec(mapped,basis,rel,3,3);mju_rotVecQuat(offset,offsets[i].position.data(),targets[i].quaternion.data());
        for(int a=0;a<3;a++)targets[i].position[a]=root[a]+scales[i]*mapped[a]+offset[a];
    }
    gmr.SetTargets(targets);
    if(cameraTracking && (input.location_flags[0]&3)==3){
        TrackedPose camera;double aligned[4];
        // Absolute HMD goal in the fixed calibrated scene/STAGE transform.
        double headDelta[3],headMapped[3];mju_sub3(headDelta,input.head.position.data(),cameraHeadOrigin.position.data());
        mju_mulMatVec(headMapped,basis,headDelta,3,3);
        for(int a=0;a<3;a++)camera.position[a]=cameraOrigin.position[a]+cameraScale*headMapped[a];
        camera.position[2]=cameraOrigin.position[2]+cameraScale*(input.head.position[1]-cameraHeadOrigin.position[1]);
        mju_mulQuat(aligned,basisQuat,input.head.quaternion.data());
        mju_mulQuat(camera.quaternion.data(),aligned,cameraRotationOffset.data());gmr.SetCameraTarget(camera);
    }else gmr.ClearCameraTarget();
    gmr.Solve();error=gmr.error;
    const double* q=gmr.data()->qpos;double delta[35]={},worldVel[3]={},localVel[3]={},R[9];mju_quat2Mat(R,q+3);
    double dt=lastTime?(input.body.time_ns-lastTime)*1e-9:0;
    if(dt>0 && dt<=.2){mj_differentiatePos(gmr.model(),delta,dt,lastQ.data(),q);std::copy_n(delta,3,worldVel);mju_mulMatTVec(localVel,R,worldVel,3,3);}
    mimic[0]=localVel[0];mimic[1]=localVel[1];mimic[2]=q[2];
    double w=q[3],x=q[4],y=q[5],z=q[6];mimic[3]=std::atan2(2*(w*x+y*z),1-2*(x*x+y*y));mimic[4]=std::asin(std::clamp(2*(w*y-z*x),-1.,1.));mimic[5]=delta[5];
    for(int i=0;i<29;i++)mimic[6+i]=q[7+i];
    std::copy_n(q,36,lastQ.begin());lastTime=input.body.time_ns;return mimic;
}

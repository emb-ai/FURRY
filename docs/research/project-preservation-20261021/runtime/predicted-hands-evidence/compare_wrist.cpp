#include "meta_retarget.h"
#include "simulation.h"
#include "episode_manifest.h"
#include "camera_pose_servo.h"
#include <fstream>
#include <sstream>
#include <map>
#include <iostream>
#include <algorithm>
#include <cmath>
static std::vector<double> Row(const std::string& line){std::stringstream s(line);std::string item;std::vector<double> row;while(std::getline(s,item,',')){double v=std::stod(item);if(!std::isfinite(v))throw std::runtime_error("Nonfinite recording");row.push_back(v);}return row;}
int main(int argc,char**argv){
    if(argc!=4)return 2;
    try{
        std::string folder=argv[2];auto manifest=episodereplay::LoadManifest(folder);
        Simulation sim(argv[1],2,manifest.scene,manifest.Policy(argv[1],folder),manifest.policyHz);manifest.ValidateDimensions(sim.model->nq,sim.model->nv,sim.model->nu);MetaRetargeter meta(argv[1]);meta.EnableVirtualLegLift(manifest.inputPatch=="virtual_leg_lift10_v1");meta.EnableNeutralWrists(manifest.neutralWrists);meta.EnableWristFrameV2(manifest.wristFrameV2);meta.EnableControllerWrists(manifest.scene=="rubik");meta.EnableControllerStageTargets(true);meta.EnableControllerAnatomicalFrame(std::string(argv[3])=="anatomical");meta.EnablePredictedHandOrientation(std::string(argv[3])=="predicted");meta.SetTravelGain(manifest.travelGainPercent/100.);
        std::ifstream inputs(folder+"/input.csv"),bodies(folder+"/body.csv"),frames(folder+"/frames.csv"),commands(folder+"/mimic.csv");
        if(!inputs||!bodies||!frames||!commands)throw std::runtime_error("GMR replay requires input/body/frames/mimic streams");
        std::map<uint64_t,TrackingFrame> poses;std::string line;
        std::getline(inputs,line);while(std::getline(inputs,line)){
            auto r=Row(line);if(r.size()!=32)throw std::runtime_error("Invalid input row");
            TrackingFrame f;f.sequence=r[0];f.xr_time_ns=r[1];f.valid=r[3];for(int j=0;j<3;j++)f.location_flags[j]=r[4+j];f.hand_active={r[7]!=0,r[8]!=0};
            TrackedPose* p[]={&f.head,&f.hands[0],&f.hands[1]};for(int i=0;i<3;i++){std::copy_n(r.begin()+11+7*i,3,p[i]->position.begin());std::copy_n(r.begin()+14+7*i,4,p[i]->quaternion.begin());}poses[f.sequence]=f;
        }
        std::getline(bodies,line);while(std::getline(bodies,line)){
            auto r=Row(line);if(r.size()!=216 || !poses.count(r[0]))throw std::runtime_error("Invalid body row");auto&b=poses.at(r[0]).body;
            b.time_ns=r[1];b.supported=r[2];b.valid=r[3];b.confidence=r[4];b.skeleton_version=r[5];
            for(int i=0;i<14;i++){int a=6+15*i;b.flags[i]=r[a];for(int k=0;k<2;k++){auto&p=k?b.rest[i]:b.joints[i];std::copy_n(r.begin()+a+1+7*k,3,p.position.begin());std::copy_n(r.begin()+a+4+7*k,4,p.quaternion.begin());}}
        }
        auto focusPauses=episodereplay::PauseTimes(folder);
        auto viewChanges=episodereplay::ViewChanges(folder);size_t viewIndex=0;bool cameraTracking=manifest.firstPerson;
        int egoCamera=mj_name2id(sim.model,mjOBJ_CAMERA,"ego");
        std::cout<<"sequence,sim_s,left_goal_distance,left_ik_distance,left_y_delta,right_goal_distance,right_ik_distance,right_y_delta,left_ik_rotation_deg,left_goal_up,left_fk_up,left_physical_up,right_ik_rotation_deg,right_goal_up,right_fk_up,right_physical_up\n";
        size_t pauseIndex=0;
        std::array<float,35> origin{};double blendStart=0,maxError=0;bool wasApplying=false;int count=0,skipped=0;
        std::getline(frames,line);std::string command;std::getline(commands,command);
        while(std::getline(frames,line)){
            if(!std::getline(commands,command))throw std::runtime_error("Missing mimic frame");auto r=Row(line),c=Row(command);
            if(r.size()!=size_t(42+sim.model->nq+sim.model->nv+sim.model->nu)||c.size()!=38 || c[0]!=r[1]||c[1]!=r[2]||c[2]!=r[3])throw std::runtime_error("Mismatched frame/mimic streams");
            while(pauseIndex<focusPauses.size() && focusPauses[pauseIndex]<=r[0]){wasApplying=false;meta.Pause();pauseIndex++;}
            while(viewIndex<viewChanges.size()&&viewChanges[viewIndex].first<=r[0])cameraTracking=viewChanges[viewIndex++].second;
            meta.EnableCameraTracking(cameraTracking);
            auto it=poses.find(r[1]);if(it==poses.end())throw std::runtime_error("Missing raw frame");const auto&f=it->second;
            std::copy_n(r.begin()+42,sim.model->nq,sim.data->qpos);mj_forward(sim.model,sim.data);
            if(r[4]){meta.calibrated=false;wasApplying=false;}
            if(r[5])meta.Calibrate(sim.model,sim.data,f);
            if(!r[6]){wasApplying=false;meta.Pause();continue;}
            if(!meta.calibrated){skipped++;continue;}
            if(r[5]||!wasApplying){origin.fill(0);origin[2]=sim.data->qpos[2];blendStart=r[3];auto*gm=meta.solver().model();for(int j=1;j<gm->njnt;j++){int source=mj_name2id(sim.model,mjOBJ_JOINT,mj_id2name(gm,mjOBJ_JOINT,j));origin[gm->jnt_qposadr[j]-1]=sim.data->qpos[sim.model->jnt_qposadr[source]];}}
            auto result=meta.Solve(f);
            if(r[3]>.5){
                std::cout<<r[1]<<','<<r[3];
                const auto& camera=meta.CameraCalibrationPose();const auto& head=meta.HeadCalibrationPose();
                for(int h=0;h<2;h++){
                    auto b=meta.solver().tasks()[12+h].body;
                    double goal[3],actual[3],mapped[3],error[3];
                    const auto& goalPose=meta.solver().Targets()[12+h];double palm[3];
                    mju_rotVecQuat(palm,meta.ControllerGripOffset(h).data(),goalPose.quaternion.data());
                    for(int k=0;k<3;k++)goal[k]=goalPose.position[k]+palm[k]-camera.position[k];
                    mju_rotVecQuat(palm,meta.ControllerGripOffset(h).data(),meta.solver().data()->xquat+4*b);
                    for(int k=0;k<3;k++)actual[k]=meta.solver().data()->xpos[3*b+k]+palm[k]-camera.position[k];
                    mju_mulMatTVec(mapped,meta.StageToRobotRotation(),goal,3,3);
                    for(int k=0;k<3;k++)error[k]=head.position[k]+meta.VisualScale()*mapped[k]-f.hands[h].position[k];
                    std::cout<<','<<mju_norm3(error);
                    mju_mulMatTVec(mapped,meta.StageToRobotRotation(),actual,3,3);
                    for(int k=0;k<3;k++)error[k]=head.position[k]+meta.VisualScale()*mapped[k]-f.hands[h].position[k];
                    std::cout<<','<<mju_norm3(error)<<','<<error[1];
                }
                for(int h=0;h<2;h++){
                    const auto& target=meta.solver().Targets()[12+h];auto* gd=meta.solver().data();int b=meta.solver().tasks()[12+h].body;
                    double inverse[4],delta[4],velocity[3];mju_negQuat(inverse,target.quaternion.data());mju_mulQuat(delta,inverse,gd->xquat+4*b);mju_quat2Vel(velocity,delta,1);
                    std::cout<<','<<mju_norm3(velocity)*180/3.141592653589793;
                    double targetR[9],axis[3],stage[3];mju_quat2Mat(targetR,target.quaternion.data());
                    for(int k=0;k<3;k++)axis[k]=targetR[3*k];mju_mulMatTVec(stage,meta.StageToRobotRotation(),axis,3,3);std::cout<<','<<stage[1];
                    for(int k=0;k<3;k++)axis[k]=gd->xmat[9*b+3*k];mju_mulMatTVec(stage,meta.StageToRobotRotation(),axis,3,3);std::cout<<','<<stage[1];
                    int actual=mj_name2id(sim.model,mjOBJ_BODY,h==0?"left_wrist_yaw_link":"right_wrist_yaw_link");
                    for(int k=0;k<3;k++)axis[k]=sim.data->xmat[9*actual+3*k];mju_mulMatTVec(stage,meta.StageToRobotRotation(),axis,3,3);std::cout<<','<<stage[1];
                }
                std::cout<<'\n';
            }
            if(manifest.catchUp){auto& solver=meta.solver();result=ApplyCameraCatchUp(result,solver.HasCameraTarget()?solver.CameraTarget():solver.CameraPose(),sim.model,sim.data,egoCamera,true,manifest.catchUpV2?solver.data()->qpos+3:nullptr);}
            double u=std::clamp((r[3]-blendStart)/.5,0.,1.);
            for(int i=0;i<35;i++){result[i]=origin[i]+u*(result[i]-origin[i]);maxError=std::max(maxError,std::abs(result[i]-c[3+i]));}
            wasApplying=true;count++;
        }
        if(std::getline(commands,command))throw std::runtime_error("Extra mimic rows");
        return count>0?0:1;
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}
}

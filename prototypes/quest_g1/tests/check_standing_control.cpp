#include "simulation.h"
#include "policy_catalog.h"
#include "episode_manifest.h"
#include <iostream>
#include <cmath>
#include <filesystem>
static void Check(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
static std::array<float,35> Command(Simulation& s){
    std::array<float,35> c{};c[2]=.74f;int k=6;
    for(auto side:{"left","right"})for(auto part:{"hip_pitch","hip_roll","hip_yaw","knee","ankle_pitch","ankle_roll"}){int j=mj_name2id(s.model,mjOBJ_JOINT,(std::string(side)+"_"+part+"_joint").c_str());c[k++]=s.data->qpos[s.model->jnt_qposadr[j]];}
    for(auto part:{"yaw","roll","pitch"}){int j=mj_name2id(s.model,mjOBJ_JOINT,(std::string("waist_")+part+"_joint").c_str());c[k++]=s.data->qpos[s.model->jnt_qposadr[j]];}
    for(auto side:{"left","right"})for(auto part:{"shoulder_pitch","shoulder_roll","shoulder_yaw","elbow","wrist_roll","wrist_pitch","wrist_yaw"}){int j=mj_name2id(s.model,mjOBJ_JOINT,(std::string(side)+"_"+part+"_joint").c_str());c[k++]=s.data->qpos[s.model->jnt_qposadr[j]];}
    return c;
}
static void Run(Simulation& s,const std::array<float,35>& c,int steps){s.SetWholeBodyReference(c);for(int k=0;k<steps;k++)s.Step(false);}
int main(int argc,char**argv){try{
    Check(argc==2,"assets required");std::string assets=argv[1];auto catalog=questpolicy::Load(assets);questpolicy::Entry standing,ordinary;
    for(auto& e:catalog){if(e.id=="homie-standing")standing=e;if(e.id=="homie-v2")ordinary=e;}
    Check(!standing.id.empty()&&standing.inputPatch=="homie_stand_v1","Missing standing catalog mode");
    Check(standing.sha256==ordinary.sha256&&standing.secondarySha256==ordinary.secondarySha256,"Standing changed author weights");
    auto policy=questpolicy::VerifiedPolicy(assets,standing);Check(policy.standing,"Standing flag lost");
    Simulation sim(assets,0,"stand",policy,50);Check(sim.Standing(),"Mode not active");
    Check(sim.model->jnt_type[0]==mjJNT_FREE,"Root was fixed");
    const auto motors=bodymotors::Load(assets+"/"+bodymotors::File);bodymotors::ValidateApplied(sim.model,motors);
    auto c=Command(sim);c[0]=.6;c[1]=-.4;c[5]=1.;Run(sim,c,2500);
    Check(!grootdecoupled::Walking(sim.DecoupledCommand()),"Navigation selected Walk in standing mode");
    Check(std::hypot(sim.data->qpos[0],sim.data->qpos[1])<.12,"Standing followed the injected velocity");
    auto bad=policy;bad.secondaryPath=assets+"/policy.onnx";bool rejected=false;try{sim.SelectPolicy(bad);}catch(...){rejected=true;}
    Check(rejected&&sim.Standing(),"Failed policy switch corrupted active mode");
    sim.SelectPolicy(questpolicy::VerifiedPolicy(assets,ordinary));Check(!sim.Standing(),"Standing leaked into ordinary HOMIE");
    c=Command(sim);c[0]=.25;Run(sim,c,3000);Check(grootdecoupled::Walking(sim.DecoupledCommand())&&sim.data->qpos[0]>.15,"Ordinary HOMIE lost walking");
    double rms[2]{};
    for(int corrected=0;corrected<2;corrected++){
        Simulation s(assets,0,"stand",policy,50);c=Command(s);
        for(int k=0;k<200;k++){auto ramp=c;ramp[21]=ramp[28]=-1.1f*(k+1)/200;Run(s,ramp,10);}
        c[21]=c[28]=-1.1f;Run(s,c,1500);mj_kinematics(s.model,s.data);
        std::vector<TrackedPose> goals(14);int wrists[2],feet[2];double feet0[6],footDrift=0;
        for(int h=0;h<2;h++){
            wrists[h]=mj_name2id(s.model,mjOBJ_BODY,h?"right_wrist_yaw_link":"left_wrist_yaw_link");feet[h]=mj_name2id(s.model,mjOBJ_BODY,h?"right_ankle_roll_link":"left_ankle_roll_link");
            std::copy_n(s.data->xpos+3*wrists[h],3,goals[12+h].position.begin());std::copy_n(s.data->xquat+4*wrists[h],4,goals[12+h].quaternion.begin());std::copy_n(s.data->xpos+3*feet[h],3,feet0+3*h);
        }
        double sum=0;int samples=0;
        for(int tick=0;tick<2000;tick++){
            auto source=c;double t=tick*.01;source[19]=.045*std::sin(2*3.141592653589793*1.2*t);source[20]=.045*std::sin(2*3.141592653589793*.8*t);
            auto command=corrected?s.CorrectStandingArms(source,goals):source;Run(s,command,10);mj_kinematics(s.model,s.data);
            for(int h=0;h<2;h++){
                footDrift=std::max(footDrift,std::hypot(s.data->xpos[3*feet[h]]-feet0[3*h],s.data->xpos[3*feet[h]+1]-feet0[3*h+1]));
                if(t>3){double delta[3];mju_sub3(delta,s.data->xpos+3*wrists[h],goals[12+h].position.data());sum+=mju_dot3(delta,delta);samples++;}
            }
            Check(!grootdecoupled::Walking(s.DecoupledCommand()),"Standing switched during torso motion");
        }
        rms[corrected]=std::sqrt(sum/samples);std::cout<<"standing correction="<<corrected<<" wrist_RMS_m="<<rms[corrected]<<" max_foot_xy_m="<<footDrift<<" root_z="<<s.data->qpos[2]<<'\n';
        Check(s.data->qpos[2]>.6&&footDrift<.06,"Standing rollout lost foot stability");
    }
    Check(rms[1]<.01&&rms[1]<.65*rms[0],"Physical wrist compensation did not improve stability");
    for(auto id:{"twist2-20k","homie-v2","homie-standing","sonic-vr3","dexweave-wide-2977"})for(auto scene:{"cup","bimanual","rubik"}){
        questpolicy::RuntimePolicy p;for(auto&e:catalog)if(e.id==id)p=questpolicy::VerifiedPolicy(assets,e);
        Simulation s(assets,0,scene,p,50);
        for(unsigned seed=0;seed<128;seed++){
            s.NewSceneTrial(seed);
            for(int k=0;k<s.data->ncon;k++){
                auto& contact=s.data->contact[k];if(contact.dist>=-.001)continue;
                int a=contact.geom[0],b=contact.geom[1];if(a<0||b<0)continue;
                const char* an=mj_id2name(s.model,mjOBJ_BODY,s.model->geom_bodyid[a]);const char* bn=mj_id2name(s.model,mjOBJ_BODY,s.model->geom_bodyid[b]);
                if(!an||!bn)continue;auto object=[](const std::string& n){return n=="cup"||n.rfind("task_object_",0)==0||n.rfind("rubik_cubie_",0)==0;};
                if(object(an)!=object(bn))Check(false,std::string("Reset penetration ")+id+"/"+scene+" seed="+std::to_string(seed)+" "+an+"/"+bn+" d="+std::to_string(contact.dist));
            }
        }
        std::cout<<id<<'/'<<scene<<": 128 actual reset poses clear\n";
    }
    std::cout<<"Standing identity, navigation, original walking, physical wrists and scene resets passed\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

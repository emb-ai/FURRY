#include "simulation.h"
#include "policy_catalog.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdexcept>

static void Check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
static void Equal(const Simulation& a,const Simulation& b){
    Check(a.PolicyHz()==b.PolicyHz(),"Policy frequency differs");
    Check(a.steps==b.steps&&a.data->time==b.data->time,"Policy reset/time differs");
    for(int i=0;i<a.model->nq;i++)Check(std::abs(a.data->qpos[i]-b.data->qpos[i])<1e-10,"Policy/history/state differs");
}
int main(int argc,char** argv){
    if(argc!=2)return 2;
    try{
        std::string assets=argv[1];auto policies=questpolicy::Load(assets);
        Simulation historical(assets,2,"stand");
        Check(historical.PolicyHz()==100,"Legacy baseline frequency changed");
        bool rejectedHz=false;try{Simulation invalid(assets,2,"stand","",60);}catch(const std::exception&){rejectedHz=true;}
        Check(rejectedHz,"Unsupported explicit cadence was accepted");
        for(int hz:{50,100})for(auto name:{"stand","cup","push_t","tennis","rubik"}){
            Simulation running(assets,2,name,"",hz),control(assets,2,name,"",hz);
            for(int i=0;i<100;i++){running.Step(false);control.Step(false);}Equal(running,control);
            for(auto bad:{assets+"/missing.onnx",assets+"/scene.xml"}){
                bool rejected=false;try{running.SelectPolicy(bad);}catch(const std::exception&){rejected=true;}
                Check(rejected,"Missing/corrupt ONNX was accepted");Equal(running,control);
                for(int i=0;i<10;i++){running.Step(false);control.Step(false);}Equal(running,control);
            }
            for(const auto& entry:policies){
                if(entry.backend!=questpolicy::Twist2 && hz!=50)continue;
                auto path=questpolicy::VerifiedPolicy(assets,entry);
                running.SelectPolicy(path);Simulation fresh(assets,2,name,path,hz);Equal(running,fresh);
                Check(running.PolicyHz()==hz,"Explicit cadence changed during policy selection");
                for(int i=0;i<300;i++){running.Step(false,.6,.2);fresh.Step(false,.6,.2);}Equal(running,fresh);
                Check(running.data->qpos[2]>.35,"Policy smoke fell");
                if(std::string(name)=="tennis"&&entry.id=="twist2-20k"){
                    for(auto* state:{running.data,fresh.data}){state->userdata[TennisTask::Armed]=1;state->userdata[TennisTask::NextServe]=state->time+7;}
                    for(int i=0;i<8200;i++){running.Step(false);fresh.Step(false);}Equal(running,fresh);
                    for(int k=0;k<TennisTask::StateSize;k++)Check(running.data->userdata[k]==fresh.data->userdata[k],"Tennis replay state differs");
                    Check(running.data->userdata[TennisTask::Serves]==1,"Runtime did not launch ball");
                    Check(running.data->userdata[TennisTask::Misses]==1,"Unreturned runtime serve missing");
                }
                if(std::string(name)=="tennis"){
                    for(int hand=0;hand<2;hand++){
                        running.SelectPolicy(path);
                        TennisTask grip;grip.Initialize(running.model);
                        auto pose=grip.HoldingPose(running.model,running.data,hand);
                        std::copy(pose.p.begin(),pose.p.end(),running.data->qpos+grip.racketQa);
                        std::copy(pose.q.begin(),pose.q.end(),running.data->qpos+grip.racketQa+3);
                        mj_forward(running.model,running.data);
                        for(int i=0;i<1000;i++){
                            running.Step(false,hand==0?.8:0,hand==1?.8:0);
                            Check(running.data->userdata[TennisTask::Holder]==hand+1,"Runtime lost grip");
                            auto expected=grip.HoldingPose(running.model,running.data,hand);
                            auto actual=grip.RacketPose(running.data);
                            Check(TennisTask::Norm(TennisTask::Add(expected.p,actual.p,-1))<.01,"Runtime racket slipped from wrist");
                        }
                        std::string side=hand==0?"left":"right";
                        auto held=grip.RacketPose(running.data);
                        int fi=0;
                        for(auto finger:{"index","thumb"}){
                            int tip=mj_name2id(running.model,mjOBJ_BODY,(side+"_hand_"+finger+"_finger_tip").c_str());
                            TennisTask::Vec p{};std::copy_n(running.data->xpos+3*tip,3,p.begin());
                            auto local=TennisTask::Rotate(held,TennisTask::Add(p,held.p,-1),true);
                            double distance=TennisTask::Cylinder(local,.085,.006).distance;
                            if(distance>=.03||(fi==0?local[2]<-.005:local[2]>.005))std::cerr<<entry.id<<" "<<side<<" "<<finger<<" blade distance="<<distance<<" normal="<<local[2]<<"\n";
                            Check(distance<.03,"Runtime fingertip left the blade");
                            Check(fi==0?local[2]>=-.005:local[2]<=.005,"Runtime thumb/index crossed blade faces");
                            fi++;
                        }
                        for(auto part:{"index_0","index_1"}){
                            int joint=mj_name2id(running.model,mjOBJ_JOINT,(side+"_hand_"+part+"_joint").c_str());
                            Check(std::abs(running.data->qpos[running.model->jnt_qposadr[joint]])<.4,"Runtime index curls into a fist");
                        }
                        if(const char* output=std::getenv("G1_GRIP_TEST_OUTPUT")){
                            std::ofstream file(std::string(output)+"/"+entry.id+"-"+side+".json");
                            file.precision(17);file<<"[";
                            for(int k=0;k<running.model->nq;k++){if(k)file<<",";file<<running.data->qpos[k];}file<<"]\n";
                        }
                    }
                }
                std::cout<<name<<": "<<entry.id<<" @ "<<running.PolicyHz()<<" Hz reset and deterministic steps passed\n";
            }
        }
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

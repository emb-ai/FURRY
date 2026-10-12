#include "simulation.h"
#include "policy_catalog.h"
#include <cmath>
#include <iostream>
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
        for(int hz:{50,100})for(auto name:{"stand","cup","push_t"}){
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
                std::cout<<name<<": "<<entry.id<<" @ "<<running.PolicyHz()<<" Hz reset and deterministic steps passed\n";
            }
        }
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

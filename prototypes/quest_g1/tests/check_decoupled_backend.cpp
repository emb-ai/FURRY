#include "simulation.h"
#include "policy_catalog.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
static void Check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static void Equal(const Simulation& a,const Simulation& b){
    Check(a.Backend()==b.Backend()&&a.PolicyHz()==b.PolicyHz()&&a.steps==b.steps,"Backend/reset differs");
    for(int i=0;i<a.model->nq;i++)Check(std::abs(a.data->qpos[i]-b.data->qpos[i])<1e-10,"State/history differs");
}
int main(int argc,char**argv){if(argc!=2)return 2;try{
    using namespace grootdecoupled;
    Command command;std::array<double,29> q=Home,dq{};std::array<double,3> omega{};
    std::array<double,4> quat{1,0,0,0};std::array<float,Actions> actions{};
    auto obs=Observe(command,q,dq,omega,quat,actions);
    Check(obs[3]==.74f&&obs[12]==-1&&obs[13]==0,"Neutral observation contract");
    quat={std::sqrt(.5),std::sqrt(.5),0,0};obs=Observe(command,q,dq,omega,quat,actions);
    Check(std::abs(obs[11]+1)<1e-6&&std::abs(obs[12])<1e-6,"Projected gravity rotation");
    command.velocity={.2f,-.1f,.4f};command.torsoRpy={.1f,.2f,-.3f};
    q[4]+=.7;dq[4]=2;omega[0]=3;actions[14]=-2;
    obs=Observe(command,q,dq,omega,quat,actions);
    Check(obs[0]==.4f&&obs[1]==-.2f&&obs[2]==.2f&&obs[7]==1.5f,"Command/state scales");
    Check(std::abs(obs[17]-.7)<1e-6&&obs[46]==.1f&&obs[85]==-2,"Ankle velocity/action history lost");
    std::array<float,Observation> history{};Append(history,obs);
    Check(history[0]==0&&history[430]==.4f&&history[515]==-2,"History left zero padding/order");
    obs[0]=.8;Append(history,obs);Check(history[344]==.4f&&history[430]==.8f,"History shift order");
    command.velocity={.049f,0,0};Check(!Walking(command),"Balance threshold");
    command.velocity={.05f,0,0};Check(Walking(command),"Walk threshold");

    const std::string assets=argv[1];questpolicy::RuntimePolicy policy;
    bool found=false;for(const auto& e:questpolicy::Load(assets))if(e.backend==questpolicy::GrootDecoupled){policy=questpolicy::VerifiedPolicy(assets,e);found=true;}
    Check(found,"Missing official decoupled policies");
    Simulation a(assets,0,"stand",policy,50),b(assets,0,"stand",policy,50);
    Check(a.PolicyHz()==50&&a.Backend()==questpolicy::GrootDecoupled,"Wrong backend");
    bool rejected=false;try{Simulation bad(assets,0,"stand",policy,100);}catch(...){rejected=true;}Check(rejected,"Unsupported decoupled cadence accepted");
    for(int i=0;i<100;i++){a.Step(false);b.Step(false);}Equal(a,b);
    auto bad=policy;bad.secondaryPath=assets+"/policy.onnx";
    rejected=false;try{a.SelectPolicy(bad);}catch(...){rejected=true;}Check(rejected,"Wrong secondary action shape accepted");
    for(int i=0;i<100;i++){a.Step(false);b.Step(false);}Equal(a,b);
    a.Reset();b.Reset();
    std::array<float,35> first{};first[2]=.74f;first[0]=.2f;
    first[22]=.2f;first[29]=-.2f;first[24]=first[31]=1.2f;
    auto second=first;for(int i=6;i<18;i++)second[i]=(i%2?1:-1)*.6f;
    a.SetWholeBodyReference(first);b.SetWholeBodyReference(second);
    for(int i=0;i<3000;i++){a.Step(false);b.Step(false);}Equal(a,b);
    Check(a.data->qpos[0]>.15,"Forward policy did not move");
    a.PauseWholeBodyReference();a.Step(false);
    for(int i=0;i<20;i++)a.Step(false);
    Check(!Walking(a.DecoupledCommand()),"Pause retained navigation");
    for(const auto& scenario:{std::array<float,3>{0,0,0},std::array<float,3>{.3f,0,0},std::array<float,3>{0,0,.5f},std::array<float,3>{0,.2f,0}}){
        a.Reset();first[0]=scenario[0];first[1]=scenario[1];first[5]=scenario[2];
        for(int i=0;i<15000;i++){
            const float wave=.5f*(1-std::cos(float(a.data->time)*1.57f));first[21]=first[28]=-.5f*wave;
            a.SetWholeBodyReference(first);a.Step(false);
        }
        Check(a.data->qpos[2]>.5,"Decoupled rollout fell");
        std::cout<<"Decoupled15s command "<<scenario[0]<<','<<scenario[1]<<','<<scenario[2]<<" height="<<a.data->qpos[2]<<'\n';
    }
    a.SelectPolicy(assets+"/policy.onnx");Simulation twist(assets,0,"stand","",50);Equal(a,twist);
    a.SelectPolicy(policy);Simulation fresh(assets,0,"stand",policy,50);Equal(a,fresh);
    std::cout<<"Decoupled contract, ignored reference legs, pause, atomic load, switching and rollouts passed\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

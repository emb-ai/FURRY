#include "simulation.h"
#include "policy_catalog.h"
#include <iostream>
#include <iomanip>
int main(int argc,char**argv){if(argc!=2)return 2;questpolicy::RuntimePolicy p;for(auto&e:questpolicy::Load(argv[1]))if(e.backend==questpolicy::GrootDecoupled)p=questpolicy::VerifiedPolicy(argv[1],e);Simulation s(argv[1],0,"stand",p,50);std::array<float,35> cmd{};cmd[2]=s.data->qpos[2];int k=0;for(int j=0;j<s.model->njnt;j++)if(s.model->jnt_type[j]==mjJNT_HINGE){const char*n=mj_id2name(s.model,mjOBJ_JOINT,j);if(n&&std::string(n).find("hand_")==std::string::npos&&k<29)cmd[6+k++]=s.data->qpos[s.model->jnt_qposadr[j]];}std::cout<<std::setprecision(17);for(int i=0;i<100;i++){if(i==50)cmd[0]=.3;s.SetWholeBodyReference(cmd);for(int j=0;j<20;j++)s.Step(false);for(int j=0;j<s.model->nq;j++)std::cout<<(j?",":"")<<s.data->qpos[j];std::cout<<'\n';}}

#include "simulation.h"
#include "policy_catalog.h"
#include <iostream>
int main(int argc,char**argv){try{
 if(argc!=2)return 2;questpolicy::RuntimePolicy policy;
 for(const auto&entry:questpolicy::Load(argv[1]))if(entry.id=="homie-v2")policy=questpolicy::VerifiedPolicy(argv[1],entry);
 for(const char* scene:{"wb_dex","cup"}){
  Simulation sim(argv[1],0,scene,policy,50);
  // Logic fixture: disable external forces/contacts, so this tests only the
  // runtime height guard rather than inventing a dynamically stable squat.
  sim.model->opt.disableflags|=mjDSBL_CONTACT|mjDSBL_GRAVITY|mjDSBL_ACTUATION;
  sim.data->qpos[2]=.30;std::fill_n(sim.data->qvel,sim.model->nv,0.);mj_forward(sim.model,sim.data);
  bool threw=false;try{sim.Step(false);}catch(const std::runtime_error&e){if(std::string(e.what()).find("G1 fell")==std::string::npos)throw;threw=true;}
  if(threw!=(std::string(scene)=="cup"))throw std::runtime_error("Wrong scene-specific physics fall guard");
  if(std::string(scene)=="wb_dex"){
   sim.data->qpos[2]=.24;std::fill_n(sim.data->qvel,sim.model->nv,0.);mj_forward(sim.model,sim.data);threw=false;
   try{sim.Step(false);}catch(const std::runtime_error&e){if(std::string(e.what()).find("G1 fell")==std::string::npos)throw;threw=true;}
   if(!threw)throw std::runtime_error("Floor-level physics guard disabled");
  }
 }
 std::cout<<"WB-Dex permits 30cm squat; rejects 24cm; other scenes retain 35cm physics guard\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

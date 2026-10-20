#include "../rubik-sequences-20261015/bridge.cpp"
struct SearchSnapshot {
 mjModel* m; mjData* d; mjData* ref;
 decltype(Simulation::standing_arms) standing_arms;
 decltype(Simulation::arm_gravity) arm_gravity;
 decltype(Simulation::dex_reference) dex_reference;
 decltype(Simulation::dex_previous) dex_previous;
 decltype(Simulation::dex_velocity) dex_velocity;
 decltype(Simulation::dex_has_reference) dex_has_reference;
 decltype(Simulation::dex_has_previous) dex_has_previous;
 decltype(Simulation::decoupled_history) decoupled_history;
 decltype(Simulation::decoupled_last_action) decoupled_last_action;
 decltype(Simulation::decoupled_command) decoupled_command;
 decltype(Simulation::scene_seed) scene_seed;
 decltype(Simulation::collection) collection;
 decltype(Simulation::tennis) tennis;
 decltype(Simulation::rubik) rubik;
 decltype(Simulation::last_action) last_action;
 decltype(Simulation::target) target;
 decltype(Simulation::history) history;
 decltype(Simulation::hand_grip) hand_grip;
 decltype(Simulation::has_whole_reference) has_whole_reference;
 decltype(Simulation::whole_reference) whole_reference;
 decltype(Simulation::has_reference) has_reference;
 decltype(Simulation::arm_reference) arm_reference;
 decltype(Simulation::steps) steps;
 SearchSnapshot(Simulation* s):m(mj_copyModel(nullptr,s->model)),d(mj_copyData(nullptr,m,s->data)),ref(mj_copyData(nullptr,m,s->reference_data)) {
 standing_arms=s->standing_arms;
 arm_gravity=s->arm_gravity;
 dex_reference=s->dex_reference;
 dex_previous=s->dex_previous;
 dex_velocity=s->dex_velocity;
 dex_has_reference=s->dex_has_reference;
 dex_has_previous=s->dex_has_previous;
 decoupled_history=s->decoupled_history;
 decoupled_last_action=s->decoupled_last_action;
 decoupled_command=s->decoupled_command;
 scene_seed=s->scene_seed;
 collection=s->collection;
 tennis=s->tennis;
 rubik=s->rubik;
 last_action=s->last_action;
 target=s->target;
 history=s->history;
 hand_grip=s->hand_grip;
 has_whole_reference=s->has_whole_reference;
 whole_reference=s->whole_reference;
 has_reference=s->has_reference;
 arm_reference=s->arm_reference;
 steps=s->steps;
 }
 void Restore(Simulation*s){mj_copyModel(s->model,m);mj_copyData(s->data,s->model,d);mj_copyData(s->reference_data,s->model,ref);
 s->standing_arms=standing_arms;
 s->arm_gravity=arm_gravity;
 s->dex_reference=dex_reference;
 s->dex_previous=dex_previous;
 s->dex_velocity=dex_velocity;
 s->dex_has_reference=dex_has_reference;
 s->dex_has_previous=dex_has_previous;
 s->decoupled_history=decoupled_history;
 s->decoupled_last_action=decoupled_last_action;
 s->decoupled_command=decoupled_command;
 s->scene_seed=scene_seed;
 s->collection=collection;
 s->tennis=tennis;
 s->rubik=rubik;
 s->last_action=last_action;
 s->target=target;
 s->history=history;
 s->hand_grip=hand_grip;
 s->has_whole_reference=has_whole_reference;
 s->whole_reference=whole_reference;
 s->has_reference=has_reference;
 s->arm_reference=arm_reference;
 s->steps=steps;
 }
 ~SearchSnapshot(){mj_deleteData(ref);mj_deleteData(d);mj_deleteModel(m);}
};
extern "C" void* snapshot(void*p){return new SearchSnapshot(static_cast<Simulation*>(p));}
extern "C" void restore(void*p,void*b){static_cast<SearchSnapshot*>(b)->Restore(static_cast<Simulation*>(p));}
extern "C" void free_snapshot(void*b){delete static_cast<SearchSnapshot*>(b);}

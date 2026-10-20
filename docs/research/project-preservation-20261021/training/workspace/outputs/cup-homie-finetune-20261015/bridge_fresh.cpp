#include "bridge.cpp"
extern "C" int fresh_reset_flags(void*p){
 auto*s=static_cast<Simulation*>(p);int result=0;
 if(s->steps==0&&s->data->time==0)result|=1;
 if(std::all_of(s->decoupled_history.begin(),s->decoupled_history.end(),[](float v){return v==0;}))result|=2;
 if(std::all_of(s->decoupled_last_action.begin(),s->decoupled_last_action.end(),[](float v){return v==0;}))result|=4;
 if(s->hand_grip[0]==0&&s->hand_grip[1]==0)result|=8;
 if(!s->has_whole_reference&&!s->has_reference)result|=16;
 bool zero=true;for(int i=0;i<s->model->nv;i++)zero=zero&&s->data->qvel[i]==0;if(zero)result|=32;
 return result;
}

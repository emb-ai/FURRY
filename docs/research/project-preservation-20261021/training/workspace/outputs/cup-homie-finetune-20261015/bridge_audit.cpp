#include "bridge.cpp"
extern "C" {
void set_phase(void*p,int phase){auto*s=static_cast<Simulation*>(p);s->steps=phase;s->data->time=phase*.001;}
void debug_state(void*p,double*target,double*ctrl,double*grip){auto*s=static_cast<Simulation*>(p);std::copy(s->target.begin(),s->target.end(),target);std::copy_n(s->data->ctrl,s->model->nu,ctrl);std::copy(s->hand_grip.begin(),s->hand_grip.end(),grip);}
}

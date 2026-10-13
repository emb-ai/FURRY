#include "rubik_task.h"
extern "C" {
void* rubik_create(const mjModel* m){auto* t=new RubikTask;t->Initialize(m);return t;}
void rubik_destroy(void* task){delete static_cast<RubikTask*>(task);}
void rubik_reset(void* task,mjModel* m,mjData* d){static_cast<RubikTask*>(task)->Reset(m,d);}
void rubik_begin(void* task,mjModel* m,mjData* d,double left,double right){static_cast<RubikTask*>(task)->Begin(m,d,left,right);}
void rubik_finish(void* task,mjModel* m,mjData* d){static_cast<RubikTask*>(task)->Finish(m,d);}
int rubik_activate(void* task,mjModel* m,mjData* d,int face){return static_cast<RubikTask*>(task)->Activate(m,d,face);}
int rubik_solved(void* task,const mjData* d){return static_cast<RubikTask*>(task)->Solved(d);}
int rubik_contacts(void* task,const mjModel* m,const mjData* d,int hand){return static_cast<RubikTask*>(task)->Contacts(m,d)[hand];}
}

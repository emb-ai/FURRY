#include "rubik_task.h"
#include <iostream>
int main(int argc,char**argv){
 if(argc!=2)return 2;char error[1024];auto*m=mj_loadXML(argv[1],nullptr,error,sizeof(error));if(!m){std::cerr<<error;return 1;}auto*d=mj_makeData(m);RubikTask task;task.Initialize(m);task.requireRotationIntent=true;task.Reset(m,d);
 auto refs=[&](double t,double left,double right){double*q=d->xquat+4*task.wrist[0];q[0]=std::cos(left/2);q[1]=q[2]=0;q[3]=std::sin(left/2);q=d->xquat+4*task.wrist[1];q[0]=std::cos(right/2);q[1]=q[2]=0;q[3]=std::sin(right/2);task.ObserveGoal(d->xquat+4*task.wrist[0],d->xquat+4*task.wrist[1],t);};
 for(int k=0;k<200;k++){double t=k*.01,a=.3*std::sin(t*3);refs(t,a,a);if(task.HasRotationIntent(t))return 3;}
 for(int k=0;k<100;k++){double t=2+k*.01;refs(t,0,(t-2)*.8);}
 if(!task.HasRotationIntent(2.99)||task.HasRotationIntent(3.1))return 4;
 refs(3.,0,.99*.8);if(task.gripGuard&&task.HasRotationIntent(3.))return 7;
 refs(4,0,1);if(task.HasRotationIntent(4))return 5;
 task.Reset(m,d);if(task.HasRotationIntent(4))return 6;
 std::cout<<"Relative command rotation gate: common motion ignored, twist admitted, stale/gap/reset cleared\n";
 mj_deleteData(d);mj_deleteModel(m);return 0;
}

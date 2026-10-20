#include "collection_task.h"
#include <iostream>
#include <iomanip>
int main(int argc,char**argv){if(argc!=2)return 2;auto*m=mj_loadModel(argv[1],nullptr);if(!m)return 3;auto*d=mj_makeData(m);CollectionTask t;t.Initialize(m);std::cout<<std::setprecision(17);for(int s=0;s<10;s++){mj_resetData(m,d);t.Reset(m,d,s);int q=m->jnt_qposadr[t.joint[0]];std::cout<<s;for(int k=0;k<7;k++)std::cout<<" "<<d->qpos[q+k];std::cout<<"\n";}mj_deleteData(d);mj_deleteModel(m);}

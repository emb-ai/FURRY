#include <mujoco/mujoco.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>

int main(int argc,char** argv){
    if(argc!=2)return 2;
    char error[2048]{};auto* m=mj_loadXML(argv[1],nullptr,error,sizeof(error));
    if(!m)throw std::runtime_error(error);
    auto* single=mj_makeData(m);auto* threaded=mj_makeData(m);
    auto* pool=mju_threadPoolCreate(2);mju_bindThreadPool(threaded,pool);
    double maxError=0;
    for(int reset=0;reset<3;reset++){
        mj_resetData(m,single);mj_resetData(m,threaded);
        if(!threaded->threadpool)throw std::runtime_error("Reset lost threadpool");
        for(int i=0;i<1000;i++){
            for(int j=0;j<m->nu;j++)single->ctrl[j]=threaded->ctrl[j]=.1*std::sin(.02*i+j);
            mj_step(m,single);mj_step(m,threaded);
            for(int j=0;j<m->nq;j++){
                double e=std::abs(single->qpos[j]-threaded->qpos[j]);
                if(!std::isfinite(e)||e>1e-8)throw std::runtime_error("Threaded physics differs");
                maxError=std::max(maxError,e);
            }
        }
        for(int j=0;j<mjNWARNING;j++)if(single->warning[j].number||threaded->warning[j].number)
            throw std::runtime_error("Physics warning");
    }
    std::printf("3000 steps, 3 resets, max qpos difference %.12g\n",maxError);
    mj_deleteData(single);mj_deleteData(threaded);mj_deleteModel(m);mju_threadPoolDestroy(pool);
}

#include "simulation.h"
#include <chrono>
#include <cstdio>
int main(int argc,char** argv){
    if(argc!=2)return 2;
    Simulation sim(argv[1]);double minz=1;
    auto start=std::chrono::steady_clock::now();
    for(int i=0;i<30000;i++){sim.Step(true,(i/5000)%2);minz=std::min(minz,sim.data->qpos[2]);}
    double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::printf("Native TWIST2: sim=%.2f wall=%.3f min_height=%.3f final=(%.3f %.3f %.3f) inference=%.3fms\n",sim.data->time,elapsed,minz,sim.data->qpos[0],sim.data->qpos[1],sim.data->qpos[2],sim.inference_ms);
    return minz>.7 && std::hypot(sim.data->qpos[0],sim.data->qpos[1])<.3 ? 0 : 1;
}

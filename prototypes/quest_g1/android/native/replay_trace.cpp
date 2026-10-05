#include "simulation.h"
#include <fstream>
#include <sstream>
#include <cstdio>
#include <algorithm>

int main(int argc,char** argv){
    if(argc!=3)return 2;
    Simulation sim(argv[1]);
    std::ifstream file(argv[2]);if(!file)return 2;
    std::string line;std::getline(file,line);
    int rows=0;double discrepancy=0;
    while(std::getline(file,line)){
        std::stringstream input(line);std::vector<double> row;
        std::string value;while(std::getline(input,value,','))row.push_back(std::stod(value));
        if(row.size()!=65){std::fprintf(stderr,"Invalid trace row %d: %zu columns\n",rows,row.size());return 2;}
        if(row[2])sim.Reset();
        if(sim.steps!=int(row[0])){std::fprintf(stderr,"Missing trace steps at row %d\n",rows);return 2;}
        discrepancy=std::max(discrepancy,std::abs(sim.data->qpos[2]-row[39]));
        if(row[3]){std::array<float,29> reference;std::copy_n(row.begin()+6,29,reference.begin());sim.SetArmReference(reference);}
        try{for(int k=0;k<10;k++)sim.Step(false,row[4],row[5]);}
        catch(const std::exception& e){std::printf("Replay failure at %.3fs, row %d: %s, height divergence %.5fm\n",sim.data->time,rows,e.what(),discrepancy);return 1;}
        rows++;
    }
    std::printf("Replay: %d frames, final %.3fs, max height divergence %.5fm\n",rows,sim.data->time,discrepancy);
    return 0;
}

#pragma once
#include <atomic>
#include <cstdint>
namespace g1graphics {
inline std::atomic<uint64_t> contextGeneration{0};
inline uint64_t NewContext(){return ++contextGeneration;}
inline uint64_t CurrentContext(){return contextGeneration.load();}
// Old numeric handles belong to a destroyed context. Forget them without
// deleting objects of the same number that may exist in the new context.
template<class... Handles> bool ForgetStale(uint64_t& generation,Handles&... handles){
    auto current=CurrentContext();if(generation==current)return false;
    ((handles=0),...);generation=current;return true;
}
}

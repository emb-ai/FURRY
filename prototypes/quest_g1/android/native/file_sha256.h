#pragma once
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace questpolicy {
// Streaming SHA-256, also used by desktop replay to reject mismatched weights.
inline std::string FileSha256(const std::string& path) {
    constexpr uint32_t k[64]={
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::array<uint32_t,8> h{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    auto rotate=[](uint32_t x,int n){return (x>>n)|(x<<(32-n));};
    auto block=[&](const unsigned char* data){
        uint32_t w[64]{};
        for(int i=0;i<16;i++)for(int j=0;j<4;j++)w[i]=(w[i]<<8)|data[4*i+j];
        for(int i=16;i<64;i++){
            uint32_t a=w[i-15],b=w[i-2];
            w[i]=w[i-16]+(rotate(a,7)^rotate(a,18)^(a>>3))+w[i-7]+(rotate(b,17)^rotate(b,19)^(b>>10));
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],v=h[7];
        for(int i=0;i<64;i++){
            uint32_t t=v+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^((~e)&g))+k[i]+w[i];
            uint32_t u=(rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b)^(a&c)^(b&c));
            v=g;g=f;f=e;e=d+t;d=c;c=b;b=a;a=t+u;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=v;
    };
    std::ifstream input(path,std::ios::binary);
    if(!input)throw std::runtime_error("Cannot read policy: "+path);
    std::array<unsigned char,64> bytes{};uint64_t count=0;
    while(input.read(reinterpret_cast<char*>(bytes.data()),64)){block(bytes.data());count+=64;}
    if(!input.eof())throw std::runtime_error("Cannot hash policy: "+path);
    size_t n=size_t(input.gcount());count+=n;bytes[n++]=0x80;
    if(n>56){while(n<64)bytes[n++]=0;block(bytes.data());n=0;}
    while(n<56)bytes[n++]=0;
    for(int j=7;j>=0;j--)bytes[n++]=static_cast<unsigned char>((count*8)>>(8*j));
    block(bytes.data());std::ostringstream out;out<<std::hex<<std::setfill('0');
    for(auto v:h)out<<std::setw(8)<<v;
    return out.str();
}
inline void VerifySha256(const std::string& path,const std::string& expected) {
    if(FileSha256(path)!=expected)throw std::runtime_error("Policy SHA256 mismatch: "+path);
}
} // namespace questpolicy

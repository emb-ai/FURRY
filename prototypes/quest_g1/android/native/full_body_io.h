#pragma once
#include "retarget.h"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// Versioned, little-endian wire format; never dumps compiler struct padding.
// SDK poses are float32. Flags, timestamp and joint IDs are preserved verbatim.
namespace fullbody {
inline constexpr uint32_t FrameBytes=48+84*36, SkeletonBytes=16+84*32;
struct Bytes {
    std::vector<char> data;size_t at=0;
    void Put(uint64_t value,int n){for(int i=0;i<n;i++)data.push_back(char((value>>(8*i))&255));}
    uint64_t Get(int n){if(at+size_t(n)>data.size())throw std::runtime_error("Truncated full body record");uint64_t v=0;for(int i=0;i<n;i++)v|=uint64_t(uint8_t(data[at++]))<<(8*i);return v;}
    void Float(double value){float f=float(value);uint32_t bits;std::memcpy(&bits,&f,4);Put(bits,4);}
    float Float(){uint32_t bits=uint32_t(Get(4));float f;std::memcpy(&f,&bits,4);return f;}
    void Pose(const TrackedPose& p){for(double v:p.position)Float(v);for(double v:p.quaternion)Float(v);}
    TrackedPose Pose(){TrackedPose p;for(auto&v:p.position)v=Float();for(auto&v:p.quaternion)v=Float();return p;}
    void Write(std::ostream& out)const{out.write(data.data(),std::streamsize(data.size()));}
    bool Read(std::istream& in,size_t n){data.resize(n);at=0;in.read(data.data(),std::streamsize(n));if(in.gcount()==0&&in.eof())return false;if(size_t(in.gcount())!=n)throw std::runtime_error("Truncated full body stream");return true;}
};
inline void Header(std::ostream& out,const char* magic,uint32_t bytes){out.write(magic,8);Bytes b;b.Put(1,4);b.Put(bytes,4);b.Put(84,4);b.Write(out);}
inline void CheckHeader(std::istream& in,const char* magic,uint32_t bytes){Bytes b;if(!b.Read(in,20)||std::memcmp(b.data.data(),magic,8))throw std::runtime_error("Invalid full body header");b.at=8;if(b.Get(4)!=1||b.Get(4)!=bytes||b.Get(4)!=84)throw std::runtime_error("Unsupported full body schema");}
class Recorder {
    std::ofstream frames,skeletons;
    std::shared_ptr<const FullBodySkeleton> lastSkeleton;
public:
    void Start(const std::filesystem::path& directory){
        lastSkeleton.reset();
        for(auto* s:{&frames,&skeletons})s->exceptions(std::ios::badbit|std::ios::failbit);
        frames.open(directory/"body_full.bin",std::ios::binary);skeletons.open(directory/"body_skeleton.bin",std::ios::binary);
        Header(frames,"G1BODY84",FrameBytes);Header(skeletons,"G1SKEL84",SkeletonBytes);
    }
    void Input(const TrackingFrame& f,int64_t received){
        const auto* full=f.body.full.get();
        if(full&&full->skeleton&&full->skeleton!=lastSkeleton){
            Bytes b;b.Put(f.sequence,8);b.Put(full->skeleton->version,4);b.Put(0,4);
            for(int j=0;j<84;j++){b.Put(uint32_t(full->skeleton->parent[j]),4);b.Pose(full->skeleton->rest[j]);}
            b.Write(skeletons);lastSkeleton=full->skeleton;
        }
        Bytes b;b.data.reserve(FrameBytes);
        b.Put(f.sequence,8);b.Put(uint64_t(received),8);b.Put(uint64_t(f.xr_time_ns),8);b.Put(uint64_t(f.body.time_ns),8);
        b.Put(full?full->version:f.body.skeleton_version,4);b.Float(f.body.confidence);
        b.Put(full?1:0,1);b.Put(full&&full->active?1:0,1);b.Put(full&&full->skeleton?1:0,1);b.Put(f.body.supported?1:0,1);b.Put(full?84:0,4);
        for(int j=0;j<84;j++){b.Put(full?full->flags[j]:0,8);b.Pose(full?full->joints[j]:TrackedPose{});}
        b.Write(frames);
    }
    void Flush(){frames.flush();skeletons.flush();}
    void Close(){frames.close();skeletons.close();lastSkeleton.reset();}
    void Abort()noexcept{for(auto* s:{&frames,&skeletons}){s->exceptions(std::ios::goodbit);if(s->is_open())s->close();s->clear();}lastSkeleton.reset();}
};
// Attach the raw stream to the same sequence IDs as input.csv during replay.
class Reader {
    std::ifstream frames;
    std::map<uint64_t,std::shared_ptr<const FullBodySkeleton>> skeletons;
    bool required=false;
public:
    Reader(const std::string& folder,bool require=false):required(require){
        auto path=std::filesystem::path(folder);
        frames.open(path/"body_full.bin",std::ios::binary);
        if(!frames){if(required)throw std::runtime_error("Foot retarget replay requires body_full.bin");return;}
        CheckHeader(frames,"G1BODY84",FrameBytes);
        std::ifstream input(path/"body_skeleton.bin",std::ios::binary);CheckHeader(input,"G1SKEL84",SkeletonBytes);
        Bytes b;while(b.Read(input,SkeletonBytes)){
            auto seq=b.Get(8);auto s=std::make_shared<FullBodySkeleton>();s->version=uint32_t(b.Get(4));b.Get(4);
            for(int j=0;j<84;j++){s->parent[j]=int32_t(uint32_t(b.Get(4)));s->rest[j]=b.Pose();}
            if(!skeletons.emplace(seq,s).second)throw std::runtime_error("Duplicate full skeleton sequence");
        }
    }
    void Attach(TrackingFrame& f){
        if(!frames.is_open())return;
        Bytes b;if(!b.Read(frames,FrameBytes))throw std::runtime_error("Missing full body input row");
        if(b.Get(8)!=f.sequence)throw std::runtime_error("Full body sequence mismatch");
        b.Get(8);if(int64_t(b.Get(8))!=f.xr_time_ns)throw std::runtime_error("Full body XR time mismatch");
        b.Get(8);auto full=std::make_shared<FullBodyFrame>();full->version=uint32_t(b.Get(4));b.Float();
        bool available=b.Get(1)!=0;full->active=b.Get(1)!=0;bool hasBind=b.Get(1)!=0;b.Get(1);
        auto count=b.Get(4);if(count!=(available?84u:0u))throw std::runtime_error("Full body count mismatch");
        for(int j=0;j<84;j++){full->flags[j]=b.Get(8);full->joints[j]=b.Pose();}
        if(hasBind){auto it=skeletons.upper_bound(f.sequence);if(it==skeletons.begin())throw std::runtime_error("Missing bind skeleton");--it;if(it->second->version!=full->version)throw std::runtime_error("Wrong bind skeleton version");full->skeleton=it->second;}
        f.body.full=available?full:nullptr;
    }
};
} // namespace fullbody

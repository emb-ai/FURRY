#pragma once
#include "camera_fusion.h"
#include <algorithm>
#include <cmath>
#include <vector>

// All positions and marker dimensions are metres in OpenXR STAGE.
// No robot transform, anatomical scale or per-skeleton recentering.
struct SkeletonMark {std::array<double,3> p;float radius;std::array<float,4> color;};
struct SkeletonBone {std::array<double,3> a,b;float radius;std::array<float,4> color;};
struct WorldSkeleton {std::vector<SkeletonMark> marks;std::vector<SkeletonBone> bones;};
inline WorldSkeleton BuildWorldSkeleton(const TrackingFrame& raw,double ageMs,const CameraOverlay& camera,bool showMeta=true,bool showCamera=true){
    WorldSkeleton out;
    auto finite=[](const auto& p){return std::all_of(p.begin(),p.end(),[](double v){return std::isfinite(v);});};
    const std::array<float,4> upper{.1f,.85f,1.f,.9f},lower{1.f,.6f,.15f,.65f};
    bool fresh=ageMs>=0 && ageMs<200;
    if(showMeta && fresh && raw.body.valid && std::abs(raw.xr_time_ns-raw.body.time_ns)<200000000LL){
        std::array<bool,14> valid{};
        for(int j=0;j<14;j++){
            valid[j]=(raw.body.flags[j]&3)==3 && finite(raw.body.joints[j].position);
            if(valid[j])out.marks.push_back({raw.body.joints[j].position,.008f,j>=2&&j<=7?lower:upper});
        }
        const int edges[][2]={{0,1},{0,2},{0,3},{2,4},{3,5},{4,6},{5,7},{1,8},{1,9},{8,10},{9,11},{10,12},{11,13}};
        for(auto& e:edges)if(valid[e[0]]&&valid[e[1]])out.bones.push_back({raw.body.joints[e[0]].position,raw.body.joints[e[1]].position,.002f,e[1]>=2&&e[1]<=7?lower:upper});
    }
    // Controller origins are distinct from anatomical wrists.
    if(showMeta && fresh)for(int hand=0;hand<2;hand++)if(raw.hand_active[hand] && (raw.location_flags[hand+1]&3)==3 && finite(raw.hands[hand].position))
        out.marks.push_back({raw.hands[hand].position,.005f,{.25f,1.f,.35f,.95f}});
    if(showCamera && camera.aligned && camera.ageMs>=0 && camera.ageMs<500){
        std::array<float,4> color{1.f,.15f,.7f,float(std::clamp((500-camera.ageMs)/350.,0.,1.))};
        std::array<bool,14> valid{};
        for(int j=0;j<14;j++){
            valid[j]=camera.valid[j] && finite(camera.points[j]);
            if(valid[j])out.marks.push_back({camera.points[j],.011f,color});
        }
        const int edges[][2]={{0,2},{2,4},{1,3},{3,5},{0,1},{7,8},{7,0},{8,1},{6,7},{6,8},{7,11},{11,9},{8,12},{12,10}};
        for(auto& e:edges)if(valid[e[0]]&&valid[e[1]])out.bones.push_back({camera.points[e[0]],camera.points[e[1]],.0025f,color});
    }
    return out;
}

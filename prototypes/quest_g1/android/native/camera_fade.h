#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

// Render-only, in STAGE metres. One centre-eye alpha is shared by both eyes.
namespace camerafade {
inline constexpr float InvisibleDistance=.08f, OpaqueDistance=.22f;
struct Bounds {
    std::array<float,3> minimum{},maximum{};
};
inline Bounds VertexBounds(const float* vertices,int count,int stride=6){
    Bounds result;
    result.minimum.fill(std::numeric_limits<float>::infinity());
    result.maximum.fill(-std::numeric_limits<float>::infinity());
    for(int i=0;i<count;i++)for(int k=0;k<3;k++){
        result.minimum[k]=std::min(result.minimum[k],vertices[i*stride+k]);
        result.maximum[k]=std::max(result.maximum[k],vertices[i*stride+k]);
    }
    return result;
}
// Exact point-to-OBB distance for the renderer's orthogonal TRS matrices.
// Mesh bounds come from the actual uploaded visual/collision vertices.
inline float Distance(const float* model,const Bounds& bounds,const float* eye){
    float center[3]={model[12],model[13],model[14]},length[3]{};
    for(int c=0;c<3;c++){
        if(bounds.maximum[c]<bounds.minimum[c])return std::numeric_limits<float>::infinity();
        float midpoint=(bounds.minimum[c]+bounds.maximum[c])*.5f;
        for(int r=0;r<3;r++){center[r]+=model[c*4+r]*midpoint;length[c]+=model[c*4+r]*model[c*4+r];}
        length[c]=std::sqrt(length[c]);
    }
    float squared=0;
    for(int c=0;c<3;c++){
        if(length[c]<1e-8f)return std::numeric_limits<float>::infinity();
        float projection=0;
        for(int r=0;r<3;r++)projection+=(eye[r]-center[r])*model[c*4+r]/length[c];
        float half=(bounds.maximum[c]-bounds.minimum[c])*.5f*length[c];
        float outside=std::max(0.f,std::abs(projection)-half);squared+=outside*outside;
    }
    return std::sqrt(squared);
}
inline float Opacity(float distance){
    float t=std::clamp((distance-InvisibleDistance)/(OpaqueDistance-InvisibleDistance),0.f,1.f);
    return t*t*(3-2*t);
}
inline float Alpha(const float* model,const Bounds& bounds,const float* eye){return Opacity(Distance(model,bounds,eye));}
}

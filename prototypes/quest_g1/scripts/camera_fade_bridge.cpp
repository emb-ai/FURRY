#include "camera_fade.h"
extern "C" float camera_alpha(const float* model,const float* vertices,int count,const float* eye){
    return camerafade::Alpha(model,camerafade::VertexBounds(vertices,count),eye);
}

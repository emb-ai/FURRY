#include "camera_fade.h"
#include <stdexcept>
#include <iostream>
static void Check(bool value,const char* text){if(!value)throw std::runtime_error(text);}
int main(){
    float identity[16]={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    camerafade::Bounds head{{-.1f,-.12f,-.14f},{.1f,.12f,.14f}};
    float eye[3]={0,0,0};Check(camerafade::Alpha(identity,head,eye)==0,"Inside head is opaque");
    eye[0]=.17f;Check(camerafade::Alpha(identity,head,eye)==0,"Near stereo eye is not protected");
    eye[0]=.25f;Check(std::abs(camerafade::Alpha(identity,head,eye)-.5f)<1e-5,"Surface-distance midpoint fade");
    eye[0]=.33f;Check(camerafade::Alpha(identity,head,eye)==1,"Far segment is hidden");
    // Rotated, translated and scaled scene, like first-person world anchoring.
    float world[16]={0,2,0,0,-2,0,0,0,0,0,2,0,1,3,5,1};
    eye[0]=1;eye[1]=3.35f;eye[2]=5;
    Check(std::abs(camerafade::Distance(world,head,eye)-.15f)<1e-5,"World scale/rotation distance");
    Check(std::abs(camerafade::Alpha(world,head,eye)-.5f)<1e-5,"Stage metre fade after scaling");
    float vertices[]={-.5f,-1,-2,0,0,1, .25f,2,3,0,0,1};
    auto offset=camerafade::VertexBounds(vertices,2);
    eye[0]=-.51f;eye[1]=0;eye[2]=0;
    Check(camerafade::Alpha(identity,offset,eye)==0,"Off-centre uploaded mesh bounds ignored");
    float previous=-1;
    for(int i=0;i<=100;i++){
        float alpha=camerafade::Opacity(i*.004f);
        Check(alpha>=previous&&alpha>=0&&alpha<=1,"Non-monotonic fade");previous=alpha;
    }
    auto empty=camerafade::VertexBounds(nullptr,0);
    Check(camerafade::Alpha(identity,empty,eye)==1,"Empty hidden mesh faded another component");
    std::cout<<"Camera surface distance, stereo clearance, scene transforms and smooth opacity passed\n";
}

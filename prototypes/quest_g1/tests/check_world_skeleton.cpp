#include "world_skeleton.h"
#include <stdexcept>
#include <limits>
#include <cstdio>
void Require(bool v,const char* message){if(!v)throw std::runtime_error(message);}
int main(){
    TrackingFrame raw;raw.body.valid=true;raw.body.flags.fill(3);
    for(int j=0;j<14;j++)raw.body.joints[j].position={.03*j,1.+.02*j,-.4};
    CameraSkeleton packet;packet.frame="pelvis-relative";packet.sourceMs=packet.receivedMs=1000;
    packet.pelvis={{2,1,-3},1,true};packet.joints[9]={{.4,.2,.1},1,true};
    CameraFusion mapper;auto camera=mapper.MapForDisplay(&packet,raw,1000);
    auto scene=BuildWorldSkeleton(raw,10,camera);
    Require(scene.marks.size()==16,"Missing body, optical wrist or optical pelvis");
    Require(scene.marks[12].p==raw.body.joints[12].position,"Meta wrist moved out of STAGE");
    Require(scene.marks[14].p==std::array<double,3>{2.4,1.2,-2.9},"Camera wrist recentered on Meta instead of measured pelvis");
    auto withController=raw;withController.hand_active[0]=true;withController.location_flags[1]=3;
    withController.hands[0].position={.5,1.2,-.6};
    auto fullCamera=camera;fullCamera.valid.fill(true);
    for(int j=0;j<14;j++)fullCamera.points[j]={1.+.03*j,1.+.02*j,-.4};
    auto hidden=BuildWorldSkeleton(withController,10,fullCamera,false,false);
    Require(hidden.marks.empty() && hidden.bones.empty(),"Disabled overlays still visible");
    auto metaOnly=BuildWorldSkeleton(withController,10,fullCamera,true,false);
    Require(metaOnly.marks.size()==15 && metaOnly.bones.size()==13,"Meta-only overlay missing body or controller");
    Require(std::none_of(metaOnly.marks.begin(),metaOnly.marks.end(),[](const auto& mark){return mark.color[2]==.7f;}),"Camera marks remain in Meta-only overlay");
    Require(std::none_of(metaOnly.bones.begin(),metaOnly.bones.end(),[](const auto& bone){return bone.color[2]==.7f;}),"Camera bones remain in Meta-only overlay");
    auto cameraOnly=BuildWorldSkeleton(withController,10,fullCamera,false,true);
    Require(cameraOnly.marks.size()==14 && cameraOnly.bones.size()==14,"Camera-only overlay includes Meta/controller or loses camera geometry");
    Require(std::all_of(cameraOnly.marks.begin(),cameraOnly.marks.end(),[](const auto& mark){return mark.color[2]==.7f;}),"Camera-only mark has Meta/controller color");
    Require(std::all_of(cameraOnly.bones.begin(),cameraOnly.bones.end(),[](const auto& bone){return bone.color[2]==.7f;}),"Camera-only bone has Meta/controller color");
    raw.body.joints[0].position={20,30,40};
    auto shifted=mapper.MapForDisplay(&packet,raw,1000);
    Require(shifted.points[9]==camera.points[9],"Moving Meta moves independent optical wrist");
    raw.body.valid=false;
    Require(mapper.MapForDisplay(&packet,raw,1000).valid[9],"Independent STAGE camera needs Meta body");
    raw.hand_active[0]=true;raw.location_flags[1]=3;raw.hands[0].position={.5,1.2,-.6};
    auto hands=BuildWorldSkeleton(raw,10,{});
    Require(hands.marks.size()==1 && hands.marks[0].p==raw.hands[0].position,"Tracked controller absent without body");
    Require(BuildWorldSkeleton(raw,200,{}).marks.empty(),"Stale hand still visible");
    Require(BuildWorldSkeleton(raw,-1,{}).marks.empty(),"Invalid age draws hand");
    camera.ageMs=500;Require(BuildWorldSkeleton(raw,200,camera).marks.empty(),"Expired optical points still visible");
    raw.hands[0].position[0]=std::numeric_limits<double>::quiet_NaN();
    Require(BuildWorldSkeleton(raw,10,{}).marks.empty(),"Nonfinite point reached renderer");
    packet.pelvis.confidence=0;Require(!mapper.MapForDisplay(&packet,raw,1000).aligned,"Missing absolute pelvis fabricated an alignment");
    puts("World skeleton: absolute STAGE, independent camera, overlay switches, hand validity and expiry passed");
}

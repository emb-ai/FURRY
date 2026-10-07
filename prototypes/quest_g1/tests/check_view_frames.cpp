#include "view_frames.h"
#include <cmath>
#include <stdexcept>
#include <cstdio>
int main(){
 TrackedPose p;p.position={.4,1.7,-.2};p.quaternion={std::cos(.3),0,std::sin(.3),0};auto head=PoseMatrix(p);
 auto camera=head;camera.m[12]=2;camera.m[13]=.8;camera.m[14]=3;
 auto world=EgoWorld(camera,head);XrMatrix4x4f composed;XrMatrix4x4f_Multiply(&composed,&world,&camera);
 for(int i=0;i<16;i++)if(std::abs(composed.m[i]-head.m[i])>1e-6)throw std::runtime_error("Ego camera mapping");
 XrFovf fov{-.8f,.9f,.85f,-.75f};XrMatrix4x4f projection;XrMatrix4x4f_CreateProjectionFov(&projection,GRAPHICS_OPENGL_ES,fov,.04f,50.f);
 for(float offset:{-.032f,.032f}){
  XrMatrix4x4f eye=head;
  for(int r=0;r<3;r++)eye.m[12+r]+=head.m[r]*offset;
  XrMatrix4x4f view,vp,hud;XrMatrix4x4f_InvertRigidBody(&view,&eye);XrMatrix4x4f_Multiply(&vp,&projection,&view);XrMatrix4x4f_Multiply(&hud,&vp,&head);
  XrVector4f point{.2f,.3f,-1.6f,1},projected;XrMatrix4x4f_TransformVector4f(&projected,&hud,&point);
  float expected=(projection.m[0]*(point.x-offset)-projection.m[8]*1.6f)/1.6f;
  if(std::abs(projected.x/projected.w-expected)>1e-6)throw std::runtime_error("HUD stereo frame");
 }
 auto rotation=LevelEgoRotation(camera,head);
 for(float scale:{.8f,1.f,1.4f})for(float delta:{-.5f,0.f,.4f}){
  auto moved=head;moved.m[12]+=delta;moved.m[13]-=delta/2;moved.m[14]-=delta;
  auto follow=CalibratedCameraWorld(camera,moved,rotation,scale);
  XrVector4f centre{camera.m[12],camera.m[13],camera.m[14],1},mapped;
  XrMatrix4x4f_TransformVector4f(&mapped,&follow,&centre);
  for(int i=0;i<3;i++)if(std::abs((&mapped.x)[i]-moved.m[12+i])>2e-6)throw std::runtime_error("Live HMD translation lost");
  XrVector4f up{0,0,1,0},level;XrMatrix4x4f_TransformVector4f(&level,&follow,&up);
  if(std::abs(level.y-scale)>1e-6 || std::abs(level.x)>1e-6 || std::abs(level.z)>1e-6)throw std::runtime_error("Floor tilted");
 }
 // Production keeps this calibration world fixed; head motion changes only
 // the eye view. A stationary prop must retain identical STAGE coordinates.
 auto fixed=CalibratedCameraWorld(camera,head,rotation,1.3f);
 XrVector4f prop{.8f,-.8f,.74f,1},initialProp;XrMatrix4x4f_TransformVector4f(&initialProp,&fixed,&prop);
 for(float dx:{-.5f,0.f,.5f}){
  auto moved=head;moved.m[12]+=dx;
  XrVector4f currentProp;XrMatrix4x4f_TransformVector4f(&currentProp,&fixed,&prop);
  if(currentProp.x!=initialProp.x || currentProp.y!=initialProp.y || currentProp.z!=initialProp.z)throw std::runtime_error("Prop moved in STAGE");
  XrMatrix4x4f view;XrMatrix4x4f_InvertRigidBody(&view,&moved);XrVector4f a,b;
  XrMatrix4x4f_TransformVector4f(&a,&view,&currentProp);
  XrMatrix4x4f originalView;XrMatrix4x4f_InvertRigidBody(&originalView,&head);XrMatrix4x4f_TransformVector4f(&b,&originalView,&initialProp);
  if(std::abs((a.x-b.x)+originalView.m[0]*dx)>1e-6)throw std::runtime_error("HMD translation not one-to-one in STAGE");
 }
 puts("Stereo, calibrated world, stationary props and one-to-one HMD displacement checks passed");
}

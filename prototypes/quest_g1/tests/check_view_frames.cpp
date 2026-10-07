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
 puts("Stereo panel and ego transform checks passed");
}

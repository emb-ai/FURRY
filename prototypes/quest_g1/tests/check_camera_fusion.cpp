#include "camera_fusion.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
void Require(bool v,const char*s){if(!v)throw std::runtime_error(s);}
int main(){
 CameraFusion fusion;TrackingFrame raw;raw.valid=raw.body.valid=true;raw.location_flags[0]=3;raw.body.confidence=1;
 CameraSkeleton c;c.frame="camera";c.pelvis.measured=true;c.pelvis.confidence=1;
 for(auto&j:c.joints){j.measured=true;j.confidence=1;}c.joints[9].confidence=c.joints[10].confidence=0;
 // Camera frame is a translated half-turn around X. No similarity scaling.
 auto camera=[](std::array<double,3> p){return std::array<double,3>{p[0]+1,2-p[1],3-p[2]};};
 for(int n=0;n<80;n++){
  double t=100000+n*80.;raw.xr_time_ns=raw.body.time_ns=int64_t(t*1e6);
  double x=.15*std::sin(n*.1),y=.1*std::cos(n*.15);
  raw.body.joints[0].position={x,1+y,0};raw.head.position={x,1.7+y,0};
  for(int side=0;side<2;side++){
   double s=side?-.15:.15;
   raw.body.joints[2+side].position={x+s,1+y,0};raw.body.joints[4+side].position={x+s,.6+y,.08};raw.body.joints[6+side].position={x+s,.2+y,0};raw.body.joints[8+side].position={x+(side?-.2:.2),1.5+y,0};
   for(int k=0;k<3;k++)c.joints[2*k+side].p=camera(raw.body.joints[2+2*k+side].position);
   c.joints[7+side].p=camera(raw.body.joints[8+side].position);
  }
  c.pelvis.p=camera(raw.body.joints[0].position);auto nose=raw.head.position;nose[2]+=.1;nose[1]-=.06;c.joints[6].p=camera(nose);
  c.sourceMs=c.receivedMs=t;c.sequence=n+1;fusion.Observe(raw,t);fusion.Apply(raw,&c,t);
 }
 Require(fusion.Aligned(),"Rigid fit did not converge");Require(fusion.stats.fitMm<.001,"Rigid fit wrong / reflected");
 auto display=fusion.MapForDisplay(&c,raw,c.sourceMs);
 Require(display.aligned && display.valid[4],"Mapped overlay missing");
 for(int a=0;a<3;a++)Require(std::abs(display.points[4][a]-raw.body.joints[6].position[a])<1e-7,"Overlay is not in Meta STAGE coordinates");
 Require(!display.valid[9],"Overlay invents missing wrist");
 auto expired=fusion.MapForDisplay(&c,raw,c.sourceMs+501);
 for(bool v:expired.valid)Require(!v,"Expired camera overlay still drawn");
 double t=c.sourceMs+80;c.sourceMs=c.receivedMs=t;c.sequence++;
 c.joints[4].p[2]-=.12;c.joints[2].p[2]-=.08;raw.body.time_ns=raw.xr_time_ns=int64_t(t*1e6);fusion.Observe(raw,t);
 auto moved=fusion.Apply(raw,&c,t);Require(moved.body.joints[6].position[2]>.005,"Camera leg correction absent");
 double oldLength[2],newLength[2];for(int k=0;k<2;k++){
  double a[3],b[3];mju_sub3(a,raw.body.joints[2+2*k].position.data(),raw.body.joints[4+2*k].position.data());mju_sub3(b,moved.body.joints[2+2*k].position.data(),moved.body.joints[4+2*k].position.data());oldLength[k]=mju_norm3(a);newLength[k]=mju_norm3(b);Require(std::abs(oldLength[k]-newLength[k])<1e-8,"Bone length changed");
 }
 auto stale=fusion.Apply(raw,&c,t+200);Require(fusion.stats.state==3,"Missing STALE flag");
 auto fallback=fusion.Apply(raw,&c,t+501);for(int j=0;j<14;j++)Require(fallback.body.joints[j].position==raw.body.joints[j].position,"Expired camera affects body");
 c.sequence++;c.sourceMs=c.receivedMs=t+600;c.pelvis.confidence=0;fusion.Apply(raw,&c,t+600);Require(fusion.stats.legs==0,"Invalid heartbeat repeats corrections");
 fusion.Reset();for(int n=0;n<50;n++){c.sourceMs=c.receivedMs=t+n*80;c.sequence++;c.pelvis.confidence=1;raw.body.time_ns=raw.xr_time_ns=int64_t(c.sourceMs*1e6);fusion.Observe(raw,c.sourceMs);fusion.Apply(raw,&c,c.sourceMs);}
 Require(!fusion.Aligned(),"Static shoulders silently accepted ambiguous alignment");
 auto unaligned=fusion.MapForDisplay(&c,raw,c.sourceMs);Require(!unaligned.aligned,"Uncalibrated overlay pretends to be mapped");
 puts("Camera rigid fit, bone projection, stale, expiry, invalid heartbeat and degeneracy checks passed");
}

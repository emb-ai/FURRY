#include "camera_legs2d.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
using namespace legs2d;
void Require(bool v,const char*s){if(!v)throw std::runtime_error(s);}
// Camera 3 m in front of an operator who faces it (+Z), 1.2 m high, looking
// along STAGE -Z. Columns of R are the camera axes x right, y down, z forward.
Camera MakeCamera(double yawDeg=0){
 Camera c;double a=yawDeg*M_PI/180;
 // Rotate the camera about the operator (origin) by yaw.
 V forward{-std::sin(a),0,-std::cos(a)},down{0,-1,0},right=Cross(down,forward);
 c.R={right[0],down[0],forward[0],right[1],down[1],forward[1],right[2],down[2],forward[2]};
 c.t={3*std::sin(a),1.2,3*std::cos(a)};c.fx=910;c.fy=905;c.cx=640;c.cy=360;return c;
}
Pixel Observe(const Camera&c,const V&p,double noise=0,std::mt19937*rng=nullptr){
 Pixel px;px.valid=c.Project(p,px.u,px.v);px.confidence=.9;
 if(rng){std::normal_distribution<double> n(0,noise);px.u+=n(*rng);px.v+=n(*rng);}
 return px;
}
// Leg from hip flexion (forward, deg) and knee flexion (deg), left leg.
void Pose(double hipDeg,double kneeDeg,V&hip,V&knee,V&ankle){
 hip={.1,1.0,0};double h=hipDeg*M_PI/180,k=(hipDeg-kneeDeg)*M_PI/180;
 knee=Add(hip,Mul(V{0,-std::cos(h),std::sin(h)},.42));
 ankle=Add(knee,Mul(V{0,-std::cos(k),std::sin(k)},.40));
}
LegInput Input(const Camera&c,double hipDeg,double kneeDeg,double noise=0,std::mt19937*rng=nullptr){
 LegInput in;V knee,ankle;Pose(hipDeg,kneeDeg,in.hip,knee,ankle);
 in.thigh=.42;in.shin=.40;in.lateral={1,0,0};
 // Meta's generated legs: standing straight, whatever the operator does.
 in.metaKnee=Add(in.hip,V{0,-.42,0});in.metaAnkle=Add(in.hip,V{0,-.82,0});
 in.knee=Observe(c,knee,noise,rng);in.ankle=Observe(c,ankle,noise,rng);return in;
}
double Error(const LegResult&r,double hipDeg,double kneeDeg){
 V hip,knee,ankle;Pose(hipDeg,kneeDeg,hip,knee,ankle);
 return std::max(Norm(Sub(r.knee,knee)),Norm(Sub(r.ankle,ankle)));
}
int main(){
 auto frontal=MakeCamera(),oblique=MakeCamera(40);
 // Exact pixels recover standing and lifted legs although Meta stands still.
 // Every bone has two ray-sphere roots; an oblique camera lets the anatomical
 // limits pick the right one for backward swings too. A frontal camera sees
 // a backward swing purely along its depth axis: only lifts are required.
 auto check=[](const Camera&c,double h,double k){
  auto r=SolveLeg(c,Input(c,h,k));Require(r.ok,"Leg solve failed");
  if(Error(r,h,k)>.02){std::fprintf(stderr,"hip %.0f knee %.0f camera x %.1f: error %.3f m\n",h,k,c.t[0],Error(r,h,k));throw std::runtime_error("Exact 2D keypoints did not recover the leg");}
  Require(r.kneePx<1 && r.anklePx<1,"Solution does not reproject onto the keypoints");
 };
 for(auto[h,k]:{std::pair{0.,0.},{30.,40.},{70.,80.},{90.,90.},{-15.,30.}})check(oblique,h,k);
 for(auto[h,k]:{std::pair{0.,0.},{30.,40.},{70.,80.},{90.,90.}})check(frontal,h,k);
 // Missing ankle: no leg, never a guess.
 {auto in=Input(frontal,30,40);in.ankle.valid=false;Require(!SolveLeg(frontal,in).ok,"Missing ankle must not drive a leg");}
 // Invalid geometry is rejected.
 {auto in=Input(frontal,0,0);in.hip[0]=NAN;Require(!SolveLeg(frontal,in).ok,"Invalid hip accepted");}
 // A frontal knee lift is the hard case: forward and backward thighs
 // foreshorten alike. The hip-extension limit must pick the anatomical
 // branch even when Meta says "standing straight".
 {auto r=SolveLeg(frontal,Input(frontal,50,60));Require(Error(r,50,60)<.02,"Frontal knee lift resolved to the wrong branch");
  Require(r.flipGap>0,"Flip gap must report the rejected branch");}
 // Noisy keypoints (5 px): bounded error, worse on the frontal camera's depth axis.
 std::mt19937 rng(7);double worst[2]={0,0};int n=0;
 for(int trial=0;trial<200;trial++){
  double h=std::uniform_real_distribution<double>(0,80)(rng),k=std::uniform_real_distribution<double>(0,90)(rng);
  for(int i=0;i<2;i++){const Camera&c=i?oblique:frontal;auto in=Input(c,h,k,5,&rng);auto r=SolveLeg(c,in);Require(r.ok,"Noisy solve failed");worst[i]+=Error(r,h,k);}n++;
 }
 std::printf("mean max-joint error at 5 px: frontal %.1f cm, oblique %.1f cm\n",100*worst[0]/n,100*worst[1]/n);
 // Regression bounds from the synthetic fixture, not accuracy claims.
 Require(worst[1]/n<.08,"Oblique camera error too large");
 Require(worst[0]/n<.16,"Frontal camera error too large");
 // Temporal prior: a previous solution keeps a nearly ambiguous leg on its branch.
 {auto in=Input(frontal,20,30);V hip,knee,ankle;Pose(20,30,hip,knee,ankle);in.hasPrevious=true;in.previousKnee=knee;in.previousAnkle=ankle;
  auto r=SolveLeg(frontal,in);Require(Error(r,20,30)<.02,"Previous solution did not hold the branch");}
 std::puts("camera 2D legs: ok");
}

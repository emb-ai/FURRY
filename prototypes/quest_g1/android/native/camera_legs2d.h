#pragma once
// Leg directions from 2D camera keypoints, without camera depth.
//
// The hip is anchored by Meta (headset/controllers) and the bone lengths are
// Meta's. Knee and ankle must reproject onto the detector's pixels; that
// fixes each bone up to a forward/backward flip and leaves the camera's depth
// axis weakly observed. The depth axis is regularised, never fitted: a weak
// pull to Meta's time-matched knee/ankle depth and to the previous solution,
// plus a soft knee hinge (no backward bend) and a soft hip-extension limit. Candidates start from both
// ray-sphere roots of each bone, Meta and the previous frame; the lowest
// cost wins, so a lift toward the camera is not locked into the wrong branch
// by the first frame's initialisation.
#include <array>
#include <algorithm>
#include <cmath>
#include <vector>

namespace legs2d {
using V=std::array<double,3>;
inline V Sub(V a,const V&b){for(int i=0;i<3;i++)a[i]-=b[i];return a;}
inline V Add(V a,const V&b){for(int i=0;i<3;i++)a[i]+=b[i];return a;}
inline V Mul(V a,double s){for(auto&x:a)x*=s;return a;}
inline double Dot(const V&a,const V&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
inline V Cross(const V&a,const V&b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
inline double Norm(const V&a){return std::sqrt(Dot(a,a));}
inline V Unit(const V&a){return Mul(a,1/std::max(1e-12,Norm(a)));}

// Pinhole camera placed in STAGE: p_stage = R p_camera + t (R row-major).
// Camera axes are OpenCV: x right, y down, z forward.
struct Camera{
 std::array<double,9> R{1,0,0,0,1,0,0,0,1};V t{};
 double fx=1,fy=1,cx=0,cy=0;
 V ToCamera(const V&p)const{V d=Sub(p,t);return {R[0]*d[0]+R[3]*d[1]+R[6]*d[2],R[1]*d[0]+R[4]*d[1]+R[7]*d[2],R[2]*d[0]+R[5]*d[1]+R[8]*d[2]};}
 V ToStage(const V&c)const{return Add({R[0]*c[0]+R[1]*c[1]+R[2]*c[2],R[3]*c[0]+R[4]*c[1]+R[5]*c[2],R[6]*c[0]+R[7]*c[1]+R[8]*c[2]},t);}
 // Unit STAGE direction of the ray through pixel (u, v).
 V Ray(double u,double v)const{return Unit(Sub(ToStage({(u-cx)/fx,(v-cy)/fy,1}),t));}
 bool Project(const V&p,double&u,double&v)const{V c=ToCamera(p);if(c[2]<.05)return false;u=fx*c[0]/c[2]+cx;v=fy*c[1]/c[2]+cy;return true;}
 double Depth(const V&p)const{return ToCamera(p)[2];}
};

struct Pixel{double u=0,v=0,confidence=0;bool valid=false;};

struct Options{
 double pixelSigma=6;      // detector keypoint noise, px at confidence 1
 double metaDepthSigma=1;  // tie-breaker toward Meta's depth along the camera axis, m
 double previousDepthSigma=.08; // temporal pull of the depth axis, m
 double hingeSigmaDeg=4;   // soft penalty on backward knee bend
 // Thigh behind vertical beyond this is anatomically rare: it is the mirror
 // of a frontal knee lift, which the 2D view alone cannot tell apart.
 double maxHipExtensionDeg=15,hipExtensionSigmaDeg=3;
 // Each bone has two ray-sphere roots for any camera placement; the wrong
 // root of an oblique view leaves the sagittal plane. The knee is a hinge
 // (varus/valgus within a few degrees) and the thigh rarely leaves the
 // sagittal plane by more than ~30 deg during teleoperation.
 double maxVarusDeg=5,varusSigmaDeg=3,maxThighOutDeg=30,thighOutSigmaDeg=5;
 int iterations=12;
};

struct LegInput{
 V hip{},metaKnee{},metaAnkle{};
 double thigh=.4,shin=.4;  // Meta bone lengths, m
 V lateral{1,0,0};         // operator's left, horizontal (knee hinge axis)
 Pixel knee,ankle;
 bool hasPrevious=false;V previousKnee{},previousAnkle{};
};

struct LegResult{
 bool ok=false;
 V thighDir{},shinDir{},knee{},ankle{};
 double cost=0,kneePx=-1,anklePx=-1;
 // Cost of the best solution whose knee lies on the other ray-sphere branch
 // (or -1 when the knee ray has a single root). A small gap means the 2D
 // view could not tell forward from backward; only the regulariser decided.
 double flipGap=-1;
};

namespace detail{
// Knee flexion angle (positive bends the shin back) about the hinge axis.
inline double Flexion(const V&thigh,const V&shin,const V&lateral){
 V axis=Sub(lateral,Mul(thigh,Dot(lateral,thigh)));if(Norm(axis)<1e-6)return 0;axis=Unit(axis);
 V back=Cross(axis,thigh);return std::atan2(Dot(shin,back),Dot(shin,thigh));
}
// Shin angle out of the knee's hinge plane.
inline double Varus(const V&thigh,const V&shin,const V&lateral){
 V axis=Sub(lateral,Mul(thigh,Dot(lateral,thigh)));if(Norm(axis)<1e-6)return 0;axis=Unit(axis);
 return std::asin(std::clamp(Dot(shin,axis),-1.,1.));
}
// Thigh angle behind vertical in the sagittal plane (STAGE Y up).
inline double HipExtension(const V&thigh,const V&lateral){
 V up{0,1,0},forward=Cross(lateral,up);if(Norm(forward)<1e-6)return 0;forward=Unit(forward);
 return std::atan2(-Dot(thigh,forward),-Dot(thigh,up));
}
// Residual vector for directions (thigh, shin).
inline int Residuals(const Camera&cam,const LegInput&in,const Options&o,const V&d1,const V&d2,double*r){
 V knee=Add(in.hip,Mul(d1,in.thigh)),ankle=Add(knee,Mul(d2,in.shin));int n=0;
 auto pixel=[&](const V&p,const Pixel&px){
  if(!px.valid){r[n++]=0;r[n++]=0;return;}
  double u,v,w=std::sqrt(std::max(px.confidence,1e-3))/o.pixelSigma;
  if(!cam.Project(p,u,v)){r[n++]=1e3;r[n++]=1e3;return;}
  r[n++]=w*(u-px.u);r[n++]=w*(v-px.v);
 };
 pixel(knee,in.knee);pixel(ankle,in.ankle);
 r[n++]=(cam.Depth(knee)-cam.Depth(in.metaKnee))/o.metaDepthSigma;
 r[n++]=(cam.Depth(ankle)-cam.Depth(in.metaAnkle))/o.metaDepthSigma;
 if(in.hasPrevious){
  r[n++]=(cam.Depth(knee)-cam.Depth(in.previousKnee))/o.previousDepthSigma;
  r[n++]=(cam.Depth(ankle)-cam.Depth(in.previousAnkle))/o.previousDepthSigma;
 }else{r[n++]=0;r[n++]=0;}
 double flex=Flexion(d1,d2,in.lateral)*180/M_PI;
 r[n++]=flex<0?flex/o.hingeSigmaDeg:0;
 r[n++]=std::max(0.,HipExtension(d1,in.lateral)*180/M_PI-o.maxHipExtensionDeg)/o.hipExtensionSigmaDeg;
 r[n++]=std::max(0.,std::abs(Varus(d1,d2,in.lateral))*180/M_PI-o.maxVarusDeg)/o.varusSigmaDeg;
 V lateral=Unit(in.lateral);
 r[n++]=std::max(0.,std::abs(std::asin(std::clamp(Dot(d1,lateral),-1.,1.)))*180/M_PI-o.maxThighOutDeg)/o.thighOutSigmaDeg;
 return n;
}
constexpr int kResiduals=12;
inline double Cost(const double*r){double s=0;for(int i=0;i<kResiduals;i++)s+=r[i]*r[i];return s;}
// Tangent basis of a unit vector.
inline void Basis(const V&d,V&a,V&b){V helper=std::abs(d[0])<.9?V{1,0,0}:V{0,1,0};a=Unit(Cross(d,helper));b=Cross(d,a);}
inline V Rotate(const V&d,const V&a,const V&b,double x,double y){return Unit(Add(d,Add(Mul(a,x),Mul(b,y))));}
// Levenberg-Marquardt over the two bone directions (4 tangent parameters).
inline double Solve(const Camera&cam,const LegInput&in,const Options&o,V&d1,V&d2){
 double r[kResiduals],trial[kResiduals],lambda=1e-3;Residuals(cam,in,o,d1,d2,r);double cost=Cost(r);
 for(int iter=0;iter<o.iterations;iter++){
  V a1,b1,a2,b2;Basis(d1,a1,b1);Basis(d2,a2,b2);
  double J[kResiduals][4];const double h=1e-6;
  for(int k=0;k<4;k++){
   V e1=d1,e2=d2;
   if(k<2)e1=Rotate(d1,a1,b1,k==0?h:0,k==1?h:0);else e2=Rotate(d2,a2,b2,k==2?h:0,k==3?h:0);
   Residuals(cam,in,o,e1,e2,trial);for(int i=0;i<kResiduals;i++)J[i][k]=(trial[i]-r[i])/h;
  }
  double A[16]={},g[4]={};
  for(int i=0;i<kResiduals;i++)for(int p=0;p<4;p++){g[p]+=J[i][p]*r[i];for(int q=0;q<4;q++)A[4*p+q]+=J[i][p]*J[i][q];}
  bool improved=false;
  for(int attempt=0;attempt<6 && !improved;attempt++){
   double M[16];std::copy_n(A,16,M);for(int p=0;p<4;p++)M[5*p]+=lambda*(1+A[5*p]);
   // 4x4 Gaussian elimination with partial pivoting.
   double x[4],b[4];for(int p=0;p<4;p++)b[p]=-g[p];bool singular=false;
   for(int c=0;c<4;c++){
    int piv=c;for(int k=c+1;k<4;k++)if(std::abs(M[4*k+c])>std::abs(M[4*piv+c]))piv=k;
    if(std::abs(M[4*piv+c])<1e-15){singular=true;break;}
    if(piv!=c){for(int k=0;k<4;k++)std::swap(M[4*c+k],M[4*piv+k]);std::swap(b[c],b[piv]);}
    for(int k=c+1;k<4;k++){double f=M[4*k+c]/M[4*c+c];for(int m=c;m<4;m++)M[4*k+m]-=f*M[4*c+m];b[k]-=f*b[c];}
   }
   if(singular){lambda*=10;continue;}
   for(int c=3;c>=0;c--){double s=b[c];for(int k=c+1;k<4;k++)s-=M[4*c+k]*x[k];x[c]=s/M[4*c+c];}
   V n1=Rotate(d1,a1,b1,x[0],x[1]),n2=Rotate(d2,a2,b2,x[2],x[3]);
   Residuals(cam,in,o,n1,n2,trial);double next=Cost(trial);
   if(next<cost){d1=n1;d2=n2;std::copy_n(trial,kResiduals,r);
    bool small=cost-next<1e-9*(1+cost);cost=next;lambda=std::max(1e-7,lambda/10);improved=true;if(small)return cost;}
   else lambda*=10;
  }
  if(!improved)break;
 }
 return cost;
}
// Points where the ray from origin along dir is at distance radius from
// centre; without an intersection, the ray point closest to the centre.
inline std::vector<V> RaySphere(const V&origin,const V&dir,const V&centre,double radius){
 V oc=Sub(origin,centre);double b=Dot(oc,dir),c=Dot(oc,oc)-radius*radius,disc=b*b-c;
 if(disc<0)return {Add(origin,Mul(dir,std::max(0.,-b)))};
 double s=std::sqrt(disc);std::vector<V> out;
 for(double l:{-b-s,-b+s})if(l>0)out.push_back(Add(origin,Mul(dir,l)));
 if(out.empty())out.push_back(Add(origin,Mul(dir,std::max(0.,-b))));
 return out;
}
}

// Solve one leg. Needs both knee and ankle pixels; returns ok=false otherwise.
inline LegResult SolveLeg(const Camera&cam,const LegInput&in,const Options&o={}){
 LegResult best;
 if(!in.knee.valid || !in.ankle.valid || !(in.thigh>.05) || !(in.shin>.05))return best;
 for(const V*p:{&in.hip,&in.metaKnee,&in.metaAnkle})for(double x:*p)if(!std::isfinite(x))return best;
 // Candidate initial directions: ray-sphere roots, Meta, previous solution.
 std::vector<std::pair<V,V>> starts;
 auto kneeRoots=detail::RaySphere(cam.t,cam.Ray(in.knee.u,in.knee.v),in.hip,in.thigh);
 for(const V&k:kneeRoots){
  V d1=Unit(Sub(k,in.hip));
  for(const V&a:detail::RaySphere(cam.t,cam.Ray(in.ankle.u,in.ankle.v),k,in.shin))starts.push_back({d1,Unit(Sub(a,k))});
 }
 starts.push_back({Unit(Sub(in.metaKnee,in.hip)),Unit(Sub(in.metaAnkle,in.metaKnee))});
 if(in.hasPrevious)starts.push_back({Unit(Sub(in.previousKnee,in.hip)),Unit(Sub(in.previousAnkle,in.previousKnee))});
 // Branch of a knee: which side of the hip's closest ray point it lies on.
 V kneeRay=cam.Ray(in.knee.u,in.knee.v);double split=Dot(Sub(in.hip,cam.t),kneeRay);
 auto branch=[&](const V&knee){return Dot(Sub(knee,cam.t),kneeRay)<split;};
 double bestCost=INFINITY,otherCost[2]={INFINITY,INFINITY};
 for(auto[d1,d2]:starts){
  double cost=detail::Solve(cam,in,o,d1,d2);
  V knee=Add(in.hip,Mul(d1,in.thigh));bool side=branch(knee);
  otherCost[side]=std::min(otherCost[side],cost);
  if(cost<bestCost){bestCost=cost;best.thighDir=d1;best.shinDir=d2;}
 }
 if(!std::isfinite(bestCost))return best;
 best.ok=true;best.cost=bestCost;
 best.knee=Add(in.hip,Mul(best.thighDir,in.thigh));best.ankle=Add(best.knee,Mul(best.shinDir,in.shin));
 bool side=branch(best.knee);
 if(kneeRoots.size()==2 && std::isfinite(otherCost[!side]))best.flipGap=otherCost[!side]-bestCost;
 double u,v;
 if(cam.Project(best.knee,u,v))best.kneePx=std::hypot(u-in.knee.u,v-in.knee.v);
 if(cam.Project(best.ankle,u,v))best.anklePx=std::hypot(u-in.ankle.u,v-in.ankle.v);
 return best;
}
}

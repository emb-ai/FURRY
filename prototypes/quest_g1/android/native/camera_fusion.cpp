#include "camera_fusion.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <fstream>
namespace {
using V=std::array<double,3>;
V Sub(V a,V b){for(int i=0;i<3;i++)a[i]-=b[i];return a;}
V Add(V a,V b){for(int i=0;i<3;i++)a[i]+=b[i];return a;}
V Mul(V a,double s){for(auto&x:a)x*=s;return a;}
double Norm(V a){return std::sqrt(mju_dot3(a.data(),a.data()));}
V Unit(V a){return Mul(a,1/std::max(1e-9,Norm(a)));}
V Rotate(const std::array<double,9>&r,V a){V b;mju_mulMatVec(b.data(),r.data(),a.data(),3,3);return b;}
bool Good(const CameraJoint&j){return j.measured && j.confidence>=.55;}
void Swing(TrackedPose& pose,V before,V after){
 before=Unit(before);after=Unit(after);double cross[3],q[4],result[4];mju_cross(cross,before.data(),after.data());
 q[0]=1+mju_dot3(before.data(),after.data());std::copy_n(cross,3,q+1);
 if(mju_normalize4(q)<1e-6)return;mju_mulQuat(result,q,pose.quaternion.data());std::copy_n(result,4,pose.quaternion.begin());
}
// Knee flexion (positive bends the shin back) and varus/valgus of a shin
// direction about the hinge axis, the operator's left made normal to the thigh.
struct KneeFrame{V axis,back;bool ok=false;};
KneeFrame Knee(V thigh,V lateral){
 KneeFrame k;V axis=Sub(lateral,Mul(thigh,mju_dot3(lateral.data(),thigh.data())));
 if(Norm(axis)<.3*Norm(lateral) || Norm(lateral)<1e-3)return k;
 k.axis=Unit(axis);double back[3];mju_cross(back,k.axis.data(),thigh.data());k.back={back[0],back[1],back[2]};k.ok=true;return k;
}
// The knee is a hinge. Camera bone directions bent it 10-20 deg backwards
// (left knee, 10% of camera frames in the 2026-10-12 episode) and up to
// 11 deg sideways; the robot knee then sat on its -5 deg stop. Flexion is
// kept at least Meta's (0-3 deg), varus/valgus within maxVarusDeg. The
// clamped leg is turned about the hip to keep the hip->ankle direction:
// laying the shin along a noisy thigh doubled ankle acceleration.
constexpr double minFlexDeg=3,maxVarusDeg=5;
void Hinge(V& thigh,V& shin,V lateral,V metaThigh,V metaShin,double upper,double lower){
 KneeFrame k=Knee(thigh,lateral),m=Knee(metaThigh,lateral);if(!k.ok)return;
 double x=mju_dot3(shin.data(),thigh.data()),y=mju_dot3(shin.data(),k.back.data()),z=mju_dot3(shin.data(),k.axis.data());
 double flex=std::atan2(y,x),varus=std::asin(std::clamp(z,-1.,1.)),floor=0;
 if(m.ok)floor=std::clamp(std::atan2(mju_dot3(metaShin.data(),m.back.data()),mju_dot3(metaShin.data(),metaThigh.data())),0.,minFlexDeg*M_PI/180);
 double limit=maxVarusDeg*M_PI/180;
 if(flex>=floor && std::abs(varus)<=limit)return;
 V reach=Add(Mul(thigh,upper),Mul(shin,lower));
 flex=std::max(flex,floor);varus=std::clamp(varus,-limit,limit);
 shin=Add(Mul(Add(Mul(thigh,std::cos(flex)),Mul(k.back,std::sin(flex))),std::cos(varus)),Mul(k.axis,std::sin(varus)));
 V before=Unit(Add(Mul(thigh,upper),Mul(shin,lower))),after=Unit(reach);double cross[3],q[4];
 mju_cross(cross,before.data(),after.data());q[0]=1+mju_dot3(before.data(),after.data());std::copy_n(cross,3,q+1);
 if(mju_normalize4(q)<1e-6)return;
 V t,s;mju_rotVecQuat(t.data(),thigh.data(),q);mju_rotVecQuat(s.data(),shin.data(),q);thigh=t;shin=s;
}
// Horn's closed-form rigid fit (scale fixed to one) of paired points.
bool Horn(const std::vector<V>&from,const std::vector<V>&to,std::array<double,9>&R,V&t){
 if(from.size()<3)return false;
 V a{},b{};for(size_t i=0;i<from.size();i++){a=Add(a,from[i]);b=Add(b,to[i]);}a=Mul(a,1./from.size());b=Mul(b,1./to.size());
 double S[9]={};for(size_t i=0;i<from.size();i++){V x=Sub(from[i],a),y=Sub(to[i],b);for(int r=0;r<3;r++)for(int k=0;k<3;k++)S[3*r+k]+=x[r]*y[k];}
 double N[16]={S[0]+S[4]+S[8],S[5]-S[7],S[6]-S[2],S[1]-S[3],
 S[5]-S[7],S[0]-S[4]-S[8],S[1]+S[3],S[2]+S[6],
 S[6]-S[2],S[1]+S[3],-S[0]+S[4]-S[8],S[5]+S[7],
 S[1]-S[3],S[2]+S[6],S[5]+S[7],-S[0]-S[4]+S[8]};
 // Jacobi eigensolver for Horn's symmetric quaternion matrix, avoiding a
 // sign-sensitive power iteration at half turns.
 double Q[16]={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
 for(int iter=0;iter<80;iter++){
  int p=0,q=1;for(int r=0;r<4;r++)for(int k=r+1;k<4;k++)if(std::abs(N[4*r+k])>std::abs(N[4*p+q])){p=r;q=k;}
  if(std::abs(N[4*p+q])<1e-12)break;
  double angle=.5*std::atan2(2*N[4*p+q],N[4*q+q]-N[4*p+p]),cs=std::cos(angle),sn=std::sin(angle);
  for(int r=0;r<4;r++){double x=N[4*r+p],y=N[4*r+q];N[4*r+p]=cs*x-sn*y;N[4*r+q]=sn*x+cs*y;}
  for(int k=0;k<4;k++){double x=N[4*p+k],y=N[4*q+k];N[4*p+k]=cs*x-sn*y;N[4*q+k]=sn*x+cs*y;}
  for(int r=0;r<4;r++){double x=Q[4*r+p],y=Q[4*r+q];Q[4*r+p]=cs*x-sn*y;Q[4*r+q]=sn*x+cs*y;}
 }
 int best=0;for(int i=1;i<4;i++)if(N[5*i]>N[5*best])best=i;double quat[4];for(int i=0;i<4;i++)quat[i]=Q[4*i+best];
 if(mju_normalize4(quat)<1e-9)return false;
 mju_quat2Mat(R.data(),quat);t=Sub(b,Rotate(R,a));return true;
}
// Horn weighs camera depth like x/y, but depth noise is ~2.5x larger on
// body landmarks. Gauss-Newton steps on residuals in camera axes with depth
// down-weighted. 3 cm beat Horn on all three 2026-10-12 recordings (median
// error vs the offline fit 1.8->1.3, 2.4->1.8, 6.1->4.9 deg; max 3.9->3.0,
// 5.6->3.4, 7.2->6.1). The offline 5.5 cm let a shoulders-only window with
// little motion lock 18 deg off.
bool Refine(const std::vector<V>&from,const std::vector<V>&to,std::array<double,9>&R,V&t){
 constexpr double weight[3]={1/(.02*.02),1/(.02*.02),1/(.03*.03)};
 if(from.size()<3)return false;
 // Camera-frame model: from ~ G to + s, with G = R^T and s = -R^T t.
 std::array<double,9> G;mju_transpose(G.data(),R.data(),3,3);V s=Mul(Rotate(G,t),-1);
 for(int iter=0;iter<4;iter++){
  double A[36]={},g[6]={};
  for(size_t i=0;i<from.size();i++){
   V q=Rotate(G,to[i]),r=Sub(from[i],Add(q,s));
   // d(G to)/d(rotation) = -[q]x, d/ds = I; residual r = from - G to - s.
   double J[3][6]={{0,-q[2],q[1],1,0,0},{q[2],0,-q[0],0,1,0},{-q[1],q[0],0,0,0,1}};
   for(int a=0;a<3;a++)for(int m=0;m<6;m++){g[m]+=weight[a]*J[a][m]*r[a];for(int n=0;n<6;n++)A[6*m+n]+=weight[a]*J[a][m]*J[a][n];}
  }
  double step[6];if(mju_cholFactor(A,6,1e-9)<6)return false;mju_cholSolve(step,A,g,6);
  double q[4],dq[4]={1,0,0,0},G4[4];mju_mat2Quat(q,G.data());
  V rotation{-step[0],-step[1],-step[2]};double angle=Norm(rotation);
  if(angle>1e-12){V axis=Mul(rotation,1/angle);mju_axisAngle2Quat(dq,axis.data(),angle);}
  mju_mulQuat(G4,dq,q);mju_normalize4(G4);mju_quat2Mat(G.data(),G4);
  for(int a=0;a<3;a++)s[a]+=step[3+a];
  if(angle<1e-6)break;
 }
 mju_transpose(R.data(),G.data(),3,3);t=Mul(Rotate(R,s),-1);return true;
}
double AngleDeg(const std::array<double,9>&a,const std::array<double,9>&b){
 double trace=0;for(int i=0;i<9;i++)trace+=a[i]*b[i];
 return std::acos(std::clamp((trace-1)/2,-1.,1.))*180/M_PI;
}
// (1-w)*a + w*b for nearby rigid transforms: normalized quaternion lerp. To
// first order this averages the mapped operator points as well.
void Mix(const std::array<double,9>&Ra,const V&ta,const std::array<double,9>&Rb,const V&tb,double w,std::array<double,9>&R,V&t){
 double qa[4],qb[4],q[4];mju_mat2Quat(qa,Ra.data());mju_mat2Quat(qb,Rb.data());
 double sign=mju_dot(qa,qb,4)<0?-1:1;for(int i=0;i<4;i++)q[i]=(1-w)*qa[i]+w*sign*qb[i];
 mju_normalize4(q);mju_quat2Mat(R.data(),q);t=Add(Mul(ta,1-w),Mul(tb,w));
}
constexpr int windowSize=300,maxWindows=9,checkSize=100;constexpr size_t minVerify=20;
constexpr double inlierM=.15,minConsistent=.2,maxStoredDeg=12,maxStoredM=.25;
}
void CameraFusion::Reset(){history.clear();lastSequence=~0ull;lastApplyMs=0;target={};halfway={};correction={};weights={};activeWeights={};lastGoodMs={};stats.state=stats.legs=stats.ready=0;stats.ageMs=-1;stats.weight=0;}
void CameraFusion::ForgetRegistration(){
 fitBuffer.clear();fitRandom=1;fitCountdown=0;aligned=false;checks.clear();
 rotation=storedRotation=windowRotation={1,0,0,0,1,0,0,0,1};translation=storedTranslation=windowTranslation={};
 storedWindows=windowPackets=0;windowFit=false;moved=verified=false;registrationVersion++;
 stats.fitPackets=0;stats.fitHint=1;stats.fitMm=-1;stats.fitProgress=0;stats.fitSplitDeg=-1;stats.fitSpreadCm=0;stats.fitInliers=0;
}
void CameraFusion::Reverify(){
 // Window packets pair camera points with the old STAGE frame.
 fitBuffer.clear();fitCountdown=0;checks.clear();windowFit=false;windowPackets=0;stats.fitPackets=0;
 if(!storedWindows){ForgetRegistration();return;}
 aligned=true;verified=moved=false;Blend();stats.fitHint=3;stats.fitProgress=.9;
}
void CameraFusion::Blend(){
 if(!windowFit){rotation=storedRotation;translation=storedTranslation;return;}
 if(!storedWindows){rotation=windowRotation;translation=windowTranslation;return;}
 // A partial window counts in proportion to its length.
 double part=std::min(1.,double(windowPackets)/windowSize);
 Mix(storedRotation,storedTranslation,windowRotation,windowTranslation,part/(std::min(storedWindows,maxWindows)+part),rotation,translation);
}
void CameraFusion::CommitWindow(){
 // Disjoint windows have independent depth noise; capping the count keeps a
 // slowly drifting camera mount (a few degrees) tracked within ~10 windows.
 if(!storedWindows){storedRotation=windowRotation;storedTranslation=windowTranslation;}
 else Mix(storedRotation,storedTranslation,windowRotation,windowTranslation,1./(std::min(storedWindows,maxWindows)+1),storedRotation,storedTranslation);
 storedWindows++;windowFit=false;windowPackets=0;fitBuffer.clear();fitCountdown=0;registrationVersion++;Blend();
}
bool CameraFusion::LoadRegistration(const std::string& path){
 std::ifstream in(path);std::string tag;int windows=0;std::array<double,9> R;V t;
 if(!(in>>tag>>windows) || tag!="camera-registration-v1" || windows<1)return false;
 for(auto&x:R)if(!(in>>x))return false;
 for(auto&x:t)if(!(in>>x))return false;
 // Reject anything that is not a proper rotation.
 std::array<double,9> RRt;mju_mulMatMatT(RRt.data(),R.data(),R.data(),3,3,3);
 for(int i=0;i<9;i++)if(!std::isfinite(RRt[i]) || std::abs(RRt[i]-(i%4==0))>1e-4)return false;
 double det=R[0]*(R[4]*R[8]-R[5]*R[7])-R[1]*(R[3]*R[8]-R[5]*R[6])+R[2]*(R[3]*R[7]-R[4]*R[6]);
 if(det<.99 ||!std::isfinite(t[0]+t[1]+t[2]))return false;
 ForgetRegistration();registrationVersion--;
 storedRotation=R;storedTranslation=t;storedWindows=windows;aligned=true;Blend();
 // Awaiting confirmation from live packets (see verified).
 stats.fitHint=3;stats.fitProgress=.9;return true;
}
bool CameraFusion::SaveRegistration(const std::string& path)const{
 if(!storedWindows)return std::remove(path.c_str())==0 || errno==ENOENT;
 std::string temporary=path+".tmp";
 {std::ofstream out(temporary,std::ios::trunc);out.precision(17);
  out<<"camera-registration-v1 "<<storedWindows<<'\n';
  for(double x:storedRotation)out<<x<<' ';
  out<<'\n';
  for(double x:storedTranslation)out<<x<<' ';
  out<<'\n';
  if(!out)return false;}
 return std::rename(temporary.c_str(),path.c_str())==0;
}
void CameraFusion::Observe(const TrackingFrame&f,double ms){
 if(!f.body.valid)return;
 // Body timestamp is returned by the runtime, unlike the predicted HMD time.
 ms+=(f.body.time_ns-f.xr_time_ns)*1e-6;
 if(!history.empty() && ms<=history.back().ms)return;
 history.push_back({ms,f});while(history.size()>300 || (!history.empty() && ms-history.front().ms>3000))history.pop_front();
}
void CameraFusion::Fit(const CameraSkeleton& c,const TrackingFrame& f){
 if(c.frame!="camera")return;
 if(!Good(c.joints[7]) || !Good(c.joints[8])){if(!aligned)stats.fitHint=1;return;}
 // Side-on shoulders put the depth sample on the near arm or the silhouette
 // edge (2026-10-12 episode: residual doubles beyond ~35 deg). Camera axes:
 // x right, z forward, so a frontal shoulder line is mostly along x.
 V across=Sub(c.joints[7].p,c.joints[8].p);
 if(std::abs(across[2])>.7*std::abs(across[0])){if(!aligned)stats.fitHint=2;return;}
 // Use homologous shoulder landmarks, not nose == HMD origin. Across a short
 // translation phase the paired shoulders supply a non-collinear rigid fit.
 FitPacket packet;
 for(int side=0;side<2;side++){packet.from[packet.n]=c.joints[7+side].p;packet.to[packet.n++]=f.body.joints[8+side].position;}
 // Optional camera wrists match Meta wrist landmarks, never Touch origins.
 for(int side=0;side<2;side++)if(Good(c.joints[9+side])){packet.from[packet.n]=c.joints[9+side].p;packet.to[packet.n++]=f.body.joints[12+side].position;}
 // A moved camera (or a Guardian redone between launches) leaves the stored
 // registration consistent with almost no packets. With the right one ~90%
 // pass; 100 packets (~9 s) outlast a second person walking through view.
 if(aligned){
  bool in=true;for(int k=0;k<packet.n && in;k++)in=Norm(Sub(Add(Rotate(rotation,packet.from[k]),translation),packet.to[k]))<=inlierM;
  checks.push_back(in);if(checks.size()>checkSize)checks.pop_front();
  size_t consistent=std::count(checks.begin(),checks.end(),1);
  if(!verified && checks.size()>=minVerify && consistent>=.6*checks.size()){verified=true;stats.fitHint=0;stats.fitProgress=1;}
  if(checks.size()>=checkSize && consistent<minConsistent*checkSize){ForgetRegistration();moved=true;}
 }
 fitBuffer.push_back(packet);if(fitBuffer.size()>windowSize)fitBuffer.pop_front();
 windowPackets++;stats.fitPackets=int(fitBuffer.size());
 // After locking, keep refining from the growing buffer at a lower rate.
 if(aligned && --fitCountdown>0)return;
 fitCountdown=10;
 constexpr size_t minPackets=40;constexpr double maxRmsM=.09,maxSplitDeg=5,minSpreadM=.06,minInliers=.5;
 if(fitBuffer.size()<minPackets){if(!aligned){stats.fitProgress=.5*fitBuffer.size()/minPackets;stats.fitHint=3;}return;}
 std::vector<V> from,to;std::vector<int> owner;
 for(size_t i=0;i<fitBuffer.size();i++)for(int k=0;k<fitBuffer[i].n;k++){from.push_back(fitBuffer[i].from[k]);to.push_back(fitBuffer[i].to[k]);owner.push_back(int(i));}
 std::array<double,9> R;V t;
 // Packets whose every landmark lies within inlierM of the fit. A second
 // person in view moves the camera skeleton without moving Meta and fails.
 auto inliers=[&](const std::array<double,9>&R,const V&t){
  std::vector<char> keep(fitBuffer.size(),1);
  for(size_t i=0;i<from.size();i++)if(Norm(Sub(Add(Rotate(R,from[i]),t),to[i]))>inlierM)keep[owner[i]]=0;
  return keep;
 };
 auto subset=[&](const std::vector<char>&keep,size_t first,size_t last,std::vector<V>&a,std::vector<V>&b){
  a.clear();b.clear();for(size_t i=0;i<from.size();i++)if(keep[owner[i]] && size_t(owner[i])>=first && size_t(owner[i])<last){a.push_back(from[i]);b.push_back(to[i]);}
 };
 // RANSAC over whole packets. Trimming the worst points cannot recover when
 // the wrong person fills most of the buffer; a consensus set can.
 // Hypotheses are scored on every third packet (~1 ms per packet otherwise,
 // inside the physics tick); the winner gets the full inlier set.
 auto score=[&](const std::array<double,9>&R,const V&t){
  size_t count=0;
  for(size_t i=0;i<fitBuffer.size();i+=3){bool in=true;for(int k=0;k<fitBuffer[i].n && in;k++)in=Norm(Sub(Add(Rotate(R,fitBuffer[i].from[k]),t),fitBuffer[i].to[k]))<=inlierM;count+=in;}
  return count;
 };
 std::array<double,9> bestR;V bestT;size_t bestScore=0;
 auto random=[&]{fitRandom^=fitRandom<<13;fitRandom^=fitRandom>>17;fitRandom^=fitRandom<<5;return fitRandom;};
 for(int iter=0;iter<64;iter++){
  size_t pick[3];for(auto&p:pick)p=random()%fitBuffer.size();
  if(pick[0]==pick[1] || pick[1]==pick[2] || pick[0]==pick[2])continue;
  std::vector<V> a,b;for(auto p:pick)for(int k=0;k<fitBuffer[p].n;k++){a.push_back(fitBuffer[p].from[k]);b.push_back(fitBuffer[p].to[k]);}
  if(!Horn(a,b,R,t))continue;
  size_t count=score(R,t);
  if(count>bestScore){bestScore=count;bestR=R;bestT=t;}
 }
 std::vector<char> best;size_t bestCount=0;
 if(bestScore){best=inliers(bestR,bestT);bestCount=std::count(best.begin(),best.end(),1);}
 if(bestCount<3){if(!aligned){stats.fitProgress=.5;stats.fitHint=4;}return;}
 std::vector<V> a,b;
 for(int iter=0;iter<3;iter++){
  subset(best,0,fitBuffer.size(),a,b);if(!Horn(a,b,R,t))return;
  auto keep=inliers(R,t);if(std::count(keep.begin(),keep.end(),1)<3)break;best=keep;
 }
 subset(best,0,fitBuffer.size(),a,b);if(!Horn(a,b,R,t))return;
 Refine(a,b,R,t);
 size_t count=std::count(best.begin(),best.end(),1);double fraction=double(count)/fitBuffer.size();
 double squared=0;V mean{};for(size_t i=0;i<a.size();i++){V e=Sub(Add(Rotate(R,a[i]),t),b[i]);squared+=mju_dot3(e.data(),e.data());mean=Add(mean,a[i]);}
 double rms=std::sqrt(squared/a.size());mean=Mul(mean,1./a.size());
 // Require motion off the shoulder line. A static pair cannot determine pitch.
 double cov[9]={};for(auto&p:a){V x=Sub(p,mean);for(int r=0;r<3;r++)for(int k=0;k<3;k++)cov[3*r+k]+=x[r]*x[k]/a.size();}
 double eig[3],ev[9],eq[4];mju_eig3(eig,ev,eq,cov);std::sort(eig,eig+3);double spread=std::sqrt(std::max(0.,eig[1]));
 // Depth noise on shoulder keypoints is ~6-9 cm RMS, so point RMS alone
 // cannot certify the rotation. Two halves of the consensus must agree.
 std::vector<size_t> order;for(size_t i=0;i<best.size();i++)if(best[i])order.push_back(i);
 double split=180;std::array<double,9> R1,R2;V t1,t2;std::vector<V> a1,b1,a2,b2;
 size_t middle=order[order.size()/2];subset(best,0,middle,a1,b1);subset(best,middle,fitBuffer.size(),a2,b2);
 if(Horn(a1,b1,R1,t1) && Horn(a2,b2,R2,t2) && Refine(a1,b1,R1,t1) && Refine(a2,b2,R2,t2))split=AngleDeg(R1,R2);
 stats.fitMm=1000*rms;stats.fitSplitDeg=split;stats.fitSpreadCm=100*spread;stats.fitInliers=fraction;
 bool ok=count>=minPackets && fraction>=minInliers && rms<maxRmsM && split<maxSplitDeg && spread>=minSpreadM;
 if(ok){
  // Disjoint windows at one camera place agree within ~7 deg. A fit far
  // from the stored average means the camera moved: never average the two.
  if(storedWindows && (AngleDeg(R,storedRotation)>maxStoredDeg || Norm(Sub(Add(Rotate(R,mean),t),Add(Rotate(storedRotation,mean),storedTranslation)))>maxStoredM)){
   storedRotation={1,0,0,0,1,0,0,0,1};storedTranslation={};storedWindows=0;registrationVersion++;
  }
  windowRotation=R;windowTranslation=t;windowFit=true;
  if(windowPackets>=windowSize)CommitWindow();else Blend();
  aligned=verified=true;moved=false;stats.fitHint=0;stats.fitProgress=1;return;
 }
 if(aligned)return;
 stats.fitProgress=std::min({1.,double(count)/minPackets,spread/minSpreadM,fraction/minInliers});
 stats.fitHint=spread<minSpreadM || count<minPackets?3:4;
 // Geometry is sufficient but the halves still disagree: not done yet.
 stats.fitProgress=std::min(stats.fitProgress,.95);
}
TrackingFrame CameraFusion::Apply(const TrackingFrame&raw,const CameraSkeleton*c,double now){
 TrackingFrame out=raw;stats.legs=0;stats.ready=0;stats.weight=0;stats.state=c?1:0;stats.ageMs=c?std::max(now-c->sourceMs,now-c->receivedMs):-1;
 double dt=lastApplyMs?std::clamp((now-lastApplyMs)/1000.,0.,.05):.01;lastApplyMs=now;
 bool fresh=c && stats.ageMs>=0 && stats.ageMs<150 && now-c->sourceMs>=-100;
 // Delayed measurements still correct the matched historical Meta pose.
 // 150 ms controls fading/status; 500 ms remains the hard expiry.
 bool usableAge=c && stats.ageMs>=0 && stats.ageMs<500 && now-c->sourceMs>=-100;
 bool newPacket=c && c->sequence!=lastSequence;
 if(newPacket){
  lastSequence=c->sequence;weights={};
  // Registration compares timestamp-matched historical poses. Its latency
  // limit need not be the strict limit for driving the current physical legs.
  bool registrationFresh=c && stats.ageMs>=0 && stats.ageMs<1000 && now-c->sourceMs>=-100;
  if(!history.empty() && registrationFresh){auto it=std::min_element(history.begin(),history.end(),[&](const Sample&a,const Sample&b){return std::abs(a.ms-c->sourceMs)<std::abs(b.ms-c->sourceMs);});
   if(std::abs(it->ms-c->sourceMs)<60){
    Fit(*c,it->frame);if(moved && !aligned && stats.fitHint>=3)stats.fitHint=5;
    bool usable=aligned && verified && c->frame=="camera";
    // pelvis-relative packets already use STAGE axes and need no camera R.
    bool stage=c->frame=="pelvis-relative";
    if(usableAge && (usable || stage) && Good(c->pelvis)){
     for(int side=0;side<2;side++){
      bool good=Good(c->joints[side]) && Good(c->joints[2+side]) && Good(c->joints[4+side]);
      if(!good){weights[side]=0;continue;}
      double conf=1;for(int k=0;k<3;k++)conf=std::min(conf,c->joints[2*k+side].confidence);
      const auto& meta=it->frame.body.joints;
      // A registered camera pelvis far from Meta's is another person (or a
      // broken registration), not a leg pose. STAGE packets carry no position.
      if(!stage && Norm(Sub(Add(Rotate(rotation,c->pelvis.p),translation),meta[0].position))>.35){weights[side]=0;continue;}
      // Transfer thigh and shin directions only, from Meta's hip with Meta's
      // bone lengths. Camera positions put knees 5-7 cm and ankles 10-14 cm
      // high (hip-midpoint pelvis, short 2D shins): that would bend standing
      // legs. Large lifts are valid, so there is no correction-size limit.
      auto direction=[&](int from,int to){V d=Sub(c->joints[to].p,c->joints[from].p);return Unit(stage?d:Rotate(rotation,d));};
      V thigh=direction(side,2+side),shin=direction(2+side,4+side);
      // Operator's left, horizontal: the knee hinge axis up to thigh roll.
      V lateral=Sub(meta[2].position,meta[3].position);lateral[1]=0;
      V metaThigh=Sub(meta[4+side].position,meta[2+side].position),metaShin=Sub(meta[6+side].position,meta[4+side].position);
      double upper=Norm(metaThigh),lower=Norm(metaShin);
      Hinge(thigh,shin,lateral,Unit(metaThigh),Unit(metaShin),upper,lower);
      V knee=Add(meta[2+side].position,Mul(thigh,upper));
      V ankle=Add(knee,Mul(shin,lower));
      std::array<V,3> delta{V{},Sub(knee,meta[4+side].position),Sub(ankle,meta[6+side].position)};
      // A leg that is not being driven starts from this packet, not from an
      // old correction swept in by the smoother.
      bool idle=activeWeights[side]<.01;
      for(int k=0;k<3;k++){target[2*k+side]=delta[k];if(idle)halfway[2*k+side]=correction[2*k+side]=delta[k];}
      weights[side]=conf;lastGoodMs[side]=c->sourceMs;
     }
    }
   }
  }
 }
 
 stats.state=(!(aligned && verified) && c && c->frame=="camera")?1:fresh?2:stats.ageMs<500 && c?3:0;
 // Packets arrive at ~12 Hz with 4-5 cm of depth noise; holding each target
 // fed the policy a 12 Hz staircase (fused knee/ankle 2nd difference 12x
 // Meta's). Two 40 ms stages still left the live reference 2-2.5x Meta's
 // jerk; 60 ms stages cut fused knee jerk 25% and ankle acceleration 30%
 // in replay (2026-10-12 episode) for ~15 ms more lag on the fused leg.
 // Speed-adaptive stages did no better: depth jumps look like steps.
 // At 3.9 m (2026-10-12 17:30) legs still shook at 1.8x Meta's acceleration;
 // 80 ms stages cut it ~20% for 11 ms more lag. Depth-axis-only stages and a
 // per-packet Kalman filter did no better: frontal leg swing is along depth
 // and the noise is a 1-4 Hz wander, not per-packet white noise.
 double smooth=1-std::exp(-dt/kCameraStageSeconds);
 for(int j=2;j<6;j++)for(int a=0;a<3;a++){halfway[j][a]+=smooth*(target[j][a]-halfway[j][a]);correction[j][a]+=smooth*(halfway[j][a]-correction[j][a]);}
 V lateral=Sub(raw.body.joints[2].position,raw.body.joints[3].position);lateral[1]=0;
 for(int side=0;side<2;side++){
  double legAge=lastGoodMs[side]?std::max(stats.ageMs,now-lastGoodMs[side]):500;
  // The correction is relative to the time-matched Meta pose, so age delays
  // the legs but must not shrink them. A 150->500 ms ramp left a 5 Hz camera
  // (250-440 ms old) at ~0.44 weight. Fade only just before the hard expiry.
  double freshness=c && legAge<500?std::clamp((500-legAge)/100.,0.,1.):0;
  if(weights[side]*freshness>.001)stats.ready++;
  // Freshness scales the output directly: filtering it as well left ~0.6
  // weight at the 500 ms expiry, which then snapped the knee by up to 19 cm.
  activeWeights[side]+=(1-std::exp(-dt/.1))*((legsEnabled?weights[side]:0)-activeWeights[side]);
  if(legAge>=500)activeWeights[side]=0;
  double w=activeWeights[side]*freshness;if(w<.001)continue;
  if(legAge>150)stats.state=3;
  int h=2+side,k=4+side,a=6+side;V hip=raw.body.joints[h].position,knee=raw.body.joints[k].position,ankle=raw.body.joints[a].position;
  V desiredKnee=Add(knee,Mul(correction[2+side],w)),desiredAnkle=Add(ankle,Mul(correction[4+side],w));
  // Chain the bones (thigh toward the desired knee, shin toward the desired
  // ankle) instead of solving the knee from the hip-ankle distance: near a
  // straight leg that solve amplified noise ~3x and flipped the bend side.
  double upper=Norm(Sub(knee,hip)),lower=Norm(Sub(ankle,knee));
  V thigh=Sub(desiredKnee,hip);if(Norm(thigh)<1e-5)continue;thigh=Unit(thigh);
  V shin=Sub(desiredAnkle,Add(hip,Mul(thigh,upper)));if(Norm(shin)<1e-5)continue;shin=Unit(shin);
  // Smoothing blends hinged targets against a moving Meta pose: hinge again.
  Hinge(thigh,shin,lateral,Unit(Sub(knee,hip)),Unit(Sub(ankle,knee)),upper,lower);
  V newKnee=Add(hip,Mul(thigh,upper)),newAnkle=Add(newKnee,Mul(shin,lower));
  Swing(out.body.joints[h],Sub(knee,hip),Sub(newKnee,hip));Swing(out.body.joints[k],Sub(ankle,knee),Sub(newAnkle,newKnee));
  out.body.joints[k].position=newKnee;out.body.joints[a].position=newAnkle;
  stats.legs++;stats.weight=std::max(stats.weight,w);
 }
 return out;
}

CameraOverlay CameraFusion::MapForDisplay(const CameraSkeleton*c,const TrackingFrame&raw,double now)const{
 CameraOverlay out;if(!c)return out;
 out.ageMs=std::max(now-c->sourceMs,now-c->receivedMs);
 bool relative=c->frame=="pelvis-relative";
 out.aligned=(aligned && c->frame=="camera") || (relative && Good(c->pelvis));
 if(!out.aligned || out.ageMs<0 || out.ageMs>=500)return out;
 for(int i=0;i<14;i++){
  const auto&j=i==13?c->pelvis:c->joints[i];if(!Good(j))continue;
  out.points[i]=relative?Add(c->pelvis.p,i==13?V{}:j.p):Add(Rotate(rotation,j.p),translation);
  out.valid[i]=true;
 }
 return out;
}

CameraOverlay CameraOverlaySmoother::Smooth(const CameraOverlay&in,double nowS){
 double dt=nowS-lastS;bool gap=lastS<0 || dt<0 || dt>.25;lastS=nowS;
 double smooth=1-std::exp(-std::clamp(dt,0.,.05)/kCameraStageSeconds);
 CameraOverlay out=in;
 for(int i=0;i<14;i++){
  // A new point, a frame gap or a 50 cm jump (another person, a new
  // registration) is not noise: snap instead of sweeping across the room.
  if(!in.aligned || !in.valid[i]){valid[i]=false;continue;}
  if(gap || !valid[i] || Norm(Sub(in.points[i],points[i]))>.5){halfway[i]=points[i]=in.points[i];valid[i]=true;continue;}
  for(int a=0;a<3;a++){halfway[i][a]+=smooth*(in.points[i][a]-halfway[i][a]);points[i][a]+=smooth*(halfway[i][a]-points[i][a]);}
  out.points[i]=points[i];
 }
 return out;
}

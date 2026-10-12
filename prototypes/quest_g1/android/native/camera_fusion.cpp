#include "camera_fusion.h"
#include <algorithm>
#include <cmath>
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
double AngleDeg(const std::array<double,9>&a,const std::array<double,9>&b){
 double trace=0;for(int i=0;i<9;i++)trace+=a[i]*b[i];
 return std::acos(std::clamp((trace-1)/2,-1.,1.))*180/M_PI;
}
}
void CameraFusion::Reset(){history.clear();fitBuffer.clear();fitRandom=1;fitCountdown=0;aligned=false;lastSequence=~0ull;lastApplyMs=0;target={};halfway={};correction={};weights={};activeWeights={};lastGoodMs={};stats={};}
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
 fitBuffer.push_back(packet);if(fitBuffer.size()>300)fitBuffer.pop_front();
 stats.fitPackets=int(fitBuffer.size());
 // After locking, keep refining from the growing buffer at a lower rate.
 if(aligned && --fitCountdown>0)return;
 fitCountdown=10;
 constexpr size_t minPackets=40;constexpr double inlierM=.15,maxRmsM=.09,maxSplitDeg=5,minSpreadM=.06,minInliers=.5;
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
 if(Horn(a1,b1,R1,t1) && Horn(a2,b2,R2,t2))split=AngleDeg(R1,R2);
 stats.fitMm=1000*rms;stats.fitSplitDeg=split;stats.fitSpreadCm=100*spread;stats.fitInliers=fraction;
 bool ok=count>=minPackets && fraction>=minInliers && rms<maxRmsM && split<maxSplitDeg && spread>=minSpreadM;
 if(ok){rotation=R;translation=t;aligned=true;stats.fitHint=0;stats.fitProgress=1;return;}
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
    Fit(*c,it->frame);
    bool usable=aligned && c->frame=="camera";
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
      V knee=Add(meta[2+side].position,Mul(thigh,Norm(Sub(meta[4+side].position,meta[2+side].position))));
      V ankle=Add(knee,Mul(shin,Norm(Sub(meta[6+side].position,meta[4+side].position))));
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
 
 stats.state=(!aligned && c && c->frame=="camera")?1:fresh?2:stats.ageMs<500 && c?3:0;
 // Packets arrive at ~12 Hz with 4-5 cm of depth noise; holding each target
 // fed the policy a 12 Hz staircase (fused knee/ankle 2nd difference 12x
 // Meta's). Two 40 ms stages bring it to Meta's level for ~40 ms more lag.
 double smooth=1-std::exp(-dt/.04);
 for(int j=2;j<6;j++)for(int a=0;a<3;a++){halfway[j][a]+=smooth*(target[j][a]-halfway[j][a]);correction[j][a]+=smooth*(halfway[j][a]-correction[j][a]);}
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
  V thigh=Sub(desiredKnee,hip);if(Norm(thigh)<1e-5)continue;
  V newKnee=Add(hip,Mul(Unit(thigh),upper));
  V shin=Sub(desiredAnkle,newKnee);if(Norm(shin)<1e-5)continue;
  V newAnkle=Add(newKnee,Mul(Unit(shin),lower));
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

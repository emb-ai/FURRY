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
}
void CameraFusion::Reset(){history.clear();fitFrom.clear();fitTo.clear();headFrom.clear();headTo.clear();aligned=false;lastSequence=~0ull;lastPacketMs=lastApplyMs=0;correction={};weights={};activeWeights={};lastGoodMs={};stats={};}
void CameraFusion::Observe(const TrackingFrame&f,double ms){
 if(!f.body.valid)return;
 // Body timestamp is returned by the runtime, unlike the predicted HMD time.
 ms+=(f.body.time_ns-f.xr_time_ns)*1e-6;
 if(!history.empty() && ms<=history.back().ms)return;
 history.push_back({ms,f});while(history.size()>300 || (!history.empty() && ms-history.front().ms>3000))history.pop_front();
}
void CameraFusion::Fit(const CameraSkeleton& c,const TrackingFrame& f){
 if(c.frame!="camera" || !Good(c.joints[7]) || !Good(c.joints[8]))return;
 // Use homologous shoulder landmarks, not nose == HMD origin. Across a short
 // translation phase the paired shoulders supply a non-collinear rigid fit.
 for(int side=0;side<2;side++){fitFrom.push_back(c.joints[7+side].p);fitTo.push_back(f.body.joints[8+side].position);}
 // Optional camera wrists match Meta wrist landmarks, never Touch origins.
 for(int side=0;side<2;side++)if(Good(c.joints[9+side])){fitFrom.push_back(c.joints[9+side].p);fitTo.push_back(f.body.joints[12+side].position);}
 // Nose and HMD have a fixed offset only while head orientation is held.
 // Centre trajectories separately: their translation offset cannot bias R.
 if(Good(c.joints[6]) && (f.location_flags[0]&3)==3){
  if(headFrom.empty())fitHeadRotation=f.head.quaternion;
  double inv[4],rel[4],angle[3];mju_negQuat(inv,fitHeadRotation.data());mju_mulQuat(rel,inv,f.head.quaternion.data());mju_quat2Vel(angle,rel,1);
  if(mju_norm3(angle)<.15){headFrom.push_back(c.joints[6].p);headTo.push_back(f.head.position);if(headFrom.size()>80){headFrom.erase(headFrom.begin());headTo.erase(headTo.begin());}}
 }
 if(fitFrom.size()>160){fitFrom.erase(fitFrom.begin(),fitFrom.begin()+2);fitTo.erase(fitTo.begin(),fitTo.begin()+2);}
 if(fitFrom.size()<40)return;
 V a{},b{};for(size_t i=0;i<fitFrom.size();i++){a=Add(a,fitFrom[i]);b=Add(b,fitTo[i]);}a=Mul(a,1./fitFrom.size());b=Mul(b,1./fitTo.size());
 double S[9]={},cov[9]={};for(size_t i=0;i<fitFrom.size();i++){V x=Sub(fitFrom[i],a),y=Sub(fitTo[i],b);for(int r=0;r<3;r++)for(int k=0;k<3;k++){S[3*r+k]+=x[r]*y[k];cov[3*r+k]+=x[r]*x[k];}}
 if(headFrom.size()>=10){
  V ca{},cb{};for(size_t i=0;i<headFrom.size();i++){ca=Add(ca,headFrom[i]);cb=Add(cb,headTo[i]);}ca=Mul(ca,1./headFrom.size());cb=Mul(cb,1./headTo.size());
  for(size_t i=0;i<headFrom.size();i++){V x=Sub(headFrom[i],ca),y=Sub(headTo[i],cb);for(int r=0;r<3;r++)for(int k=0;k<3;k++){S[3*r+k]+=x[r]*y[k];cov[3*r+k]+=x[r]*x[k];}}
 }
 // Require motion off the shoulder line. A static pair cannot determine pitch.
 double eig[3],ev[9],eq[4];mju_eig3(eig,ev,eq,cov);std::sort(eig,eig+3);
 if(eig[1]/fitFrom.size()<.0004)return;
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
 std::array<double,9> R;mju_quat2Mat(R.data(),quat);V t=Sub(b,Rotate(R,a));double squared=0;
 for(size_t i=0;i<fitFrom.size();i++){V e=Sub(Add(Rotate(R,fitFrom[i]),t),fitTo[i]);squared+=mju_dot3(e.data(),e.data());}
 stats.fitMm=1000*std::sqrt(squared/fitFrom.size());if(stats.fitMm>40)return;
 rotation=R;translation=t;aligned=true;
}
TrackingFrame CameraFusion::Apply(const TrackingFrame&raw,const CameraSkeleton*c,double now){
 TrackingFrame out=raw;stats.legs=0;stats.weight=0;stats.state=c?1:0;stats.ageMs=c?std::max(now-c->sourceMs,now-c->receivedMs):-1;
 double dt=lastApplyMs?std::clamp((now-lastApplyMs)/1000.,0.,.05):.01;lastApplyMs=now;
 bool fresh=c && stats.ageMs>=0 && stats.ageMs<150 && now-c->sourceMs>=-100;
 bool newPacket=c && c->sequence!=lastSequence;
 if(newPacket){
  lastSequence=c->sequence;weights={};
  if(!history.empty() && fresh){auto it=std::min_element(history.begin(),history.end(),[&](const Sample&a,const Sample&b){return std::abs(a.ms-c->sourceMs)<std::abs(b.ms-c->sourceMs);});
   if(std::abs(it->ms-c->sourceMs)<60){
    if(!aligned)Fit(*c,it->frame);
    bool usable=aligned && c->frame=="camera";
    // pelvis-relative packets already use STAGE axes and need no camera R.
    bool stage=c->frame=="pelvis-relative";
    if((usable || stage) && Good(c->pelvis)){
     for(int side=0;side<2;side++){
      bool good=Good(c->joints[side]) && Good(c->joints[2+side]) && Good(c->joints[4+side]);
      if(!good){weights[side]=0;continue;}
      double conf=1;for(int k=0;k<3;k++)conf=std::min(conf,c->joints[2*k+side].confidence);
      V pelvis=it->frame.body.joints[0].position;
      bool plausible=true;std::array<V,3> delta;
      for(int k=0;k<3;k++){
       V rel=stage?c->joints[2*k+side].p:Rotate(rotation,Sub(c->joints[2*k+side].p,c->pelvis.p));
       V target=Add(pelvis,rel);delta[k]=Sub(target,it->frame.body.joints[2+2*k+side].position);
       if(Norm(delta[k])>.45)plausible=false;
      }
      if(!plausible){weights[side]=0;continue;}
      double packetDt=lastPacketMs?std::clamp((c->sourceMs-lastPacketMs)/1000.,.001,.15):.08;
      double alpha=1-std::exp(-packetDt/.06);
      for(int k=0;k<3;k++)for(int a=0;a<3;a++)correction[2*k+side][a]+=alpha*(delta[k][a]-correction[2*k+side][a]);
      weights[side]=conf;lastGoodMs[side]=c->sourceMs;
     }
    }
   }
  }
  lastPacketMs=c->sourceMs;
 }
 
 stats.state=(!aligned && c && c->frame=="camera")?1:fresh?2:stats.ageMs<500 && c?3:0;
 for(int side=0;side<2;side++){
  double legAge=lastGoodMs[side]?std::max(stats.ageMs,now-lastGoodMs[side]):500;
  double freshness=c && legAge<500?std::clamp((500-legAge)/350.,0.,1.):0;
  activeWeights[side]+=(1-std::exp(-dt/.1))*(weights[side]*freshness-activeWeights[side]);
  if(legAge>=500)activeWeights[side]=0;
  double w=activeWeights[side];if(w<.001)continue;
  if(legAge>150)stats.state=3;
  int h=2+side,k=4+side,a=6+side;V hip=raw.body.joints[h].position,knee=raw.body.joints[k].position,ankle=raw.body.joints[a].position;
  V desiredKnee=Add(knee,Mul(correction[2+side],w)),desiredAnkle=Add(ankle,Mul(correction[4+side],w));
  double upper=Norm(Sub(knee,hip)),lower=Norm(Sub(ankle,knee));V direction=Unit(Sub(desiredAnkle,hip));double distance=std::clamp(Norm(Sub(desiredAnkle,hip)),std::abs(upper-lower)+.005,upper+lower-.005);
  double along=(upper*upper-lower*lower+distance*distance)/(2*distance),height=std::sqrt(std::max(0.,upper*upper-along*along));
  V bend=Sub(desiredKnee,hip);bend=Sub(bend,Mul(direction,mju_dot3(bend.data(),direction.data())));
  if(Norm(bend)<.005){bend=Sub(knee,hip);bend=Sub(bend,Mul(direction,mju_dot3(bend.data(),direction.data())));}
  if(Norm(bend)<1e-5)continue;
  V newKnee=Add(hip,Add(Mul(direction,along),Mul(Unit(bend),height))),newAnkle=Add(hip,Mul(direction,distance));
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
 out.aligned=(aligned && c->frame=="camera") || (relative && raw.body.valid);
 if(!out.aligned || out.ageMs<0 || out.ageMs>=500)return out;
 for(int i=0;i<12;i++){
  const auto&j=i==11?c->pelvis:c->joints[i];if(!Good(j))continue;
  out.points[i]=relative?Add(raw.body.joints[0].position,i==11?V{}:j.p):Add(Rotate(rotation,j.p),translation);
  out.valid[i]=true;
 }
 return out;
}

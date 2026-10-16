#include "camera_stream.h"
#include <ixwebsocket/IXWebSocket.h>
#include <json/json.h>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <sstream>
namespace {
bool Joint(const Json::Value&v,CameraJoint&j){
 if(!v.isObject() || !v["p"].isArray() || v["p"].size()!=3 || !v["conf"].isNumeric())return false;
 for(int i=0;i<3;i++){if(!v["p"][i].isNumeric())return false;j.p[i]=v["p"][i].asDouble();if(!std::isfinite(j.p[i]) || std::abs(j.p[i])>30)return false;}
 j.confidence=v["conf"].asDouble();if(!std::isfinite(j.confidence) || j.confidence<0 || j.confidence>1)return false;
 auto src=v.get("src","window").asString();j.measured=src=="window";return true;
}
Json::Value Pose(const TrackedPose&p){Json::Value v;for(auto x:p.position)v["p"].append(x);for(auto x:p.quaternion)v["q"].append(x);return v;}
}
double CameraEpochMs(){return std::chrono::duration<double,std::milli>(std::chrono::system_clock::now().time_since_epoch()).count();}
bool CameraStream::Parse(const std::string&s,double received,CameraSkeleton&out){
 if(s.size()>16384)return false;
 Json::CharReaderBuilder builder;builder["failIfExtra"]=true;builder["rejectDupKeys"]=true;Json::Value v;std::string errors;
 std::unique_ptr<Json::CharReader> reader(builder.newCharReader());if(!reader->parse(s.data(),s.data()+s.size(),&v,&errors))return false;
 if(!v.isObject() || !v["t"].isNumeric() || !v["seq"].isUInt64() || !v["frame"].isString() || !v["joints"].isObject())return false;
 CameraSkeleton c;c.sourceMs=v["t"].asDouble();c.receivedMs=received;c.sequence=v["seq"].asUInt64();c.frame=v["frame"].asString();
 if(!std::isfinite(c.sourceMs) || c.sourceMs<=0 || (c.frame!="camera" && c.frame!="pelvis-relative"))return false;
 if(!Joint(v["pelvis"],c.pelvis))return false;
 const char*names[]={"lhip","rhip","lknee","rknee","lankle","rankle","nose","lshoulder","rshoulder","lwrist","rwrist","lelbow","relbow"};
 for(int i=0;i<13;i++){if(i>=9 && !v["joints"].isMember(names[i]))continue;if(!Joint(v["joints"][names[i]],c.joints[i]))return false;}
 // Optional raw 2D keypoints with pinhole intrinsics. Malformed values
 // reject the packet; an absent name means "not detected".
 if(v.isMember("kp2d") || v.isMember("intr")){
  const auto&intr=v["intr"];const auto&kp=v["kp2d"];
  if(!intr.isArray() || intr.size()!=4 || !kp.isObject())return false;
  for(int i=0;i<4;i++){if(!intr[i].isNumeric())return false;c.intrinsics[i]=intr[i].asDouble();}
  if(!(c.intrinsics[0]>50 && c.intrinsics[0]<1e4 && c.intrinsics[1]>50 && c.intrinsics[1]<1e4) ||
     !(std::abs(c.intrinsics[2])<1e4 && std::abs(c.intrinsics[3])<1e4))return false;
  for(const auto&name:kp.getMemberNames()){
   int i=int(std::find_if(std::begin(names),std::end(names),[&](const char*n){return name==n;})-std::begin(names));
   if(i>=13)return false;
   const auto&k=kp[name];if(!k.isArray() || k.size()!=3)return false;
   for(int a=0;a<3;a++)if(!k[a].isNumeric())return false;
   auto&p=c.keypoints[i];p.u=k[0].asDouble();p.v=k[1].asDouble();p.confidence=k[2].asDouble();
   if(!std::isfinite(p.u) || !std::isfinite(p.v) || std::abs(p.u)>1e5 || std::abs(p.v)>1e5 ||
      !std::isfinite(p.confidence) || p.confidence<0 || p.confidence>1)return false;
   p.valid=true;
  }
  c.hasKeypoints=true;
 }
 // Optional PC-side AprilTag registration; must be a proper rotation.
 if(v.isMember("stage_from_camera")){
  const auto&f=v["stage_from_camera"];
  if(!f.isObject() || !f["R"].isArray() || f["R"].size()!=9 || !f["t"].isArray() || f["t"].size()!=3)return false;
  for(int i=0;i<9;i++){if(!f["R"][i].isNumeric())return false;c.stageRotation[i]=f["R"][i].asDouble();}
  for(int i=0;i<3;i++){if(!f["t"][i].isNumeric())return false;c.stageTranslation[i]=f["t"][i].asDouble();if(!std::isfinite(c.stageTranslation[i]) || std::abs(c.stageTranslation[i])>30)return false;}
  const auto&R=c.stageRotation;
  for(int r=0;r<3;r++)for(int k=0;k<3;k++){double d=R[3*r]*R[3*k]+R[3*r+1]*R[3*k+1]+R[3*r+2]*R[3*k+2];if(!std::isfinite(d) || std::abs(d-(r==k))>1e-3)return false;}
  double det=R[0]*(R[4]*R[8]-R[5]*R[7])-R[1]*(R[3]*R[8]-R[5]*R[6])+R[2]*(R[3]*R[7]-R[4]*R[6]);
  if(det<.99)return false;
  c.hasStagePose=true;
 }
 out=c;return true;
}
CameraStream::CameraStream(const std::string&url):socket(std::make_unique<ix::WebSocket>()){
 if(url.empty())return;
 socket->setUrl(url);socket->setHandshakeTimeout(3);socket->setPingInterval(5);
 socket->setPerMessageDeflateOptions(ix::WebSocketPerMessageDeflateOptions(false));
 socket->setMaxWaitBetweenReconnectionRetries(3000);
 socket->setOnMessageCallback([this](const ix::WebSocketMessagePtr&m){
  if(m->type==ix::WebSocketMessageType::Open){connected=true;clockSynced=false;clockRttMs=-1;std::lock_guard<std::mutex>g(mutex);have=false;}
  else if(m->type==ix::WebSocketMessageType::Close || m->type==ix::WebSocketMessageType::Error)connected=false;
  else if(m->type==ix::WebSocketMessageType::Message && !m->binary){
   double now=CameraEpochMs();
   try{
    Json::CharReaderBuilder b;Json::Value value;std::string errors;std::unique_ptr<Json::CharReader>r(b.newCharReader());
    if(m->str.size()<1024 && r->parse(m->str.data(),m->str.data()+m->str.size(),&value,&errors) && value.get("type","").asString()=="pong"){
     if(value["t"].isNumeric() && value["server_t"].isNumeric()){
      double start=value["t"].asDouble(),server=value["server_t"].asDouble(),rtt=now-start;
      if(std::isfinite(start) && std::isfinite(server) && rtt>=0 && rtt<500 && (clockRttMs<0 || rtt<clockRttMs)){
       double offset=(start+now)/2-server;
       std::lock_guard<std::mutex>g(mutex);if(!clockSynced || std::abs(offset-clockOffsetMs)>20)have=false;
       clockOffsetMs=offset;clockRttMs=rtt;clockSynced=true;
      }
     }
     return;
    }
   }catch(...){rejected++;return;}
   CameraSkeleton c;try{if(!Parse(m->str,CameraEpochMs(),c)){rejected++;return;}}catch(...){rejected++;return;}
   if(clockSynced)c.sourceMs+=clockOffsetMs;
   std::lock_guard<std::mutex>g(mutex);if(have && (c.sequence<=latest.sequence || c.sourceMs<=latest.sourceMs)){rejected++;return;}latest=c;latestRaw=m->str;have=true;
  }
 });
 socket->start();running=true;
 sender=std::thread([this]{double lastPing=0;while(running){double now=CameraEpochMs();if(connected && now-lastPing>2000){Json::Value ping;ping["type"]="ping";ping["t"]=now;Json::StreamWriterBuilder b;b["indentation"]="";socket->send(Json::writeString(b,ping));lastPing=now;}std::string s;{std::lock_guard<std::mutex>g(mutex);s.swap(pendingUplink);}if(connected && !s.empty() && socket->bufferedAmount()<16384)socket->send(s);std::this_thread::sleep_for(std::chrono::milliseconds(10));}});
}
CameraStream::~CameraStream(){running=false;if(sender.joinable())sender.join();socket->stop();}
bool CameraStream::Latest(CameraSkeleton&out,std::string* raw)const{std::lock_guard<std::mutex>g(mutex);if(!have)return false;out=latest;if(raw)*raw=latestRaw;return true;}
void CameraStream::Submit(const TrackingFrame&f){
 if(!connected || !f.valid)return;
 Json::Value v;v["t"]=CameraEpochMs();v["hmd"]=Pose(f.head);
 for(int side=0;side<2;side++){auto key=side?"ctrl_r":"ctrl_l";v[key]=Pose(f.hands[side]);v[key]["tracked"]=(f.location_flags[side+1]&3)==3;}
 Json::StreamWriterBuilder b;b["indentation"]="";auto encoded=Json::writeString(b,v);std::lock_guard<std::mutex>g(mutex);pendingUplink=std::move(encoded);
}

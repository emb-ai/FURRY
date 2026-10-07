#include "camera_stream.h"
#include <ixwebsocket/IXWebSocket.h>
#include <json/json.h>
#include <chrono>
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
 const char*names[]={"lhip","rhip","lknee","rknee","lankle","rankle","nose","lshoulder","rshoulder","lwrist","rwrist"};
 for(int i=0;i<11;i++){if(i>=9 && !v["joints"].isMember(names[i]))continue;if(!Joint(v["joints"][names[i]],c.joints[i]))return false;}
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

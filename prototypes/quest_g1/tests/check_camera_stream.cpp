#include "camera_stream.h"
#include <chrono>
#include <cstdio>
#include <thread>
#include <stdexcept>
#include <cmath>
int main(int argc,char**argv){
 if(argc!=2)return 2;
 CameraSkeleton c;
 for(const char* bad:{"{}","not json","{\"t\":1,\"seq\":1,\"frame\":\"unknown\"}"})if(CameraStream::Parse(bad,2,c))throw std::runtime_error("Malformed packet accepted");
 CameraStream stream(argv[1]);TrackingFrame frame;frame.valid=true;frame.location_flags={3,3,3};
 for(int i=0;i<100;i++){
  stream.Submit(frame);
  if(stream.Latest(c) && c.sequence==3){
   if(c.joints[9].p[0]!=.4 || c.joints[11].p[0]!=.3 || !c.joints[11].measured)throw std::runtime_error("Optional arm points not parsed");
   if(c.joints[12].confidence!=0)throw std::runtime_error("Missing elbow fabricated");
  }
  if(i>10 && stream.Latest(c) && stream.rejected>=2 && stream.clockSynced){if(std::abs(stream.clockOffsetMs-700)>100)throw std::runtime_error("Clock skew not corrected");if(c.sequence!=3)throw std::runtime_error("Reordered packet accepted");puts("WebSocket handshake, skeleton parsing, uplink, rejection and sequence checks passed");return 0;}
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
 }
 return 1;
}

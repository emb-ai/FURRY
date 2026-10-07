#include "camera_stream.h"
#include <chrono>
#include <cstdio>
#include <thread>
#include <stdexcept>
int main(int argc,char**argv){
 if(argc!=2)return 2;
 CameraSkeleton c;
 for(const char* bad:{"{}","not json","{\"t\":1,\"seq\":1,\"frame\":\"unknown\"}"})if(CameraStream::Parse(bad,2,c))throw std::runtime_error("Malformed packet accepted");
 CameraStream stream(argv[1]);TrackingFrame frame;frame.valid=true;frame.location_flags={3,3,3};
 for(int i=0;i<100;i++){
  stream.Submit(frame);
  if(i>10 && stream.Latest(c) && stream.rejected>=2){if(c.sequence!=2)throw std::runtime_error("Reordered packet accepted");puts("WebSocket handshake, skeleton parsing, uplink, rejection and sequence checks passed");return 0;}
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
 }
 return 1;
}

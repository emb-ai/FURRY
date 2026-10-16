#include "camera_stream.h"
#include <chrono>
#include <cstdio>
#include <thread>
#include <stdexcept>
#include <cmath>
#include <string>
int main(int argc,char**argv){
 if(argc!=2)return 2;
 CameraSkeleton c;
 for(const char* bad:{"{}","not json","{\"t\":1,\"seq\":1,\"frame\":\"unknown\"}"})if(CameraStream::Parse(bad,2,c))throw std::runtime_error("Malformed packet accepted");
 {
  // 2D keypoints, intrinsics and the AprilTag registration are optional.
  std::string joints="\"joints\":{";const char*names[]={"lhip","rhip","lknee","rknee","lankle","rankle","nose","lshoulder","rshoulder"};
  for(int i=0;i<9;i++)joints+=std::string(i?",":"")+"\""+names[i]+"\":{\"p\":[0,0,0],\"conf\":0,\"src\":\"missing\"}";
  joints+="}";
  std::string head="{\"t\":1,\"seq\":1,\"frame\":\"camera\",\"pelvis\":{\"p\":[0,0,0],\"conf\":0,\"src\":\"missing\"},"+joints;
  auto parse=[&](const std::string&tail){return CameraStream::Parse(head+tail+"}",2,c);};
  if(!parse("") || c.hasKeypoints || c.hasStagePose)throw std::runtime_error("Plain packet changed");
  if(!parse(",\"intr\":[910,905,640,360],\"kp2d\":{\"lknee\":[600.5,400,0.9]},\"stage_from_camera\":{\"R\":[1,0,0,0,-1,0,0,0,-1],\"t\":[0,1.2,3]}"))throw std::runtime_error("2D packet rejected");
  if(!c.hasKeypoints || !c.keypoints[2].valid || c.keypoints[2].u!=600.5 || c.keypoints[0].valid || c.intrinsics[3]!=360)throw std::runtime_error("2D keypoints not parsed");
  if(!c.hasStagePose || c.stageRotation[4]!=-1 || c.stageTranslation[2]!=3)throw std::runtime_error("Stage pose not parsed");
  for(const char* bad:{",\"kp2d\":{\"lknee\":[1,2,0.5]}",",\"intr\":[910,905,640,360],\"kp2d\":{\"tail\":[1,2,0.5]}",
    ",\"intr\":[910,905,640,360],\"kp2d\":{\"lknee\":[1,2,1.5]}",",\"intr\":[0,905,640,360],\"kp2d\":{}",
    ",\"stage_from_camera\":{\"R\":[2,0,0,0,1,0,0,0,1],\"t\":[0,0,0]}",",\"stage_from_camera\":{\"R\":[1,0,0,0,1,0,0,0,-1],\"t\":[0,0,0]}"})
   if(parse(bad))throw std::runtime_error("Malformed 2D packet accepted");
 }
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

#include "virtual_leg_lift.h"
#include <iostream>
static void check(bool b,const char* s){if(!b)throw std::runtime_error(s);}
static double distance(const TrackedPose&a,const TrackedPose&b){double v[3];mju_sub3(v,a.position.data(),b.position.data());return mju_norm3(v);}
int main(){try{
 BodyFrame b;b.valid=true;b.time_ns=1000000000;
 for(int i=0;i<14;i++)b.flags[i]=3;
 for(int s=0;s<2;s++){b.joints[2+s].position={s*.2,.9,0};b.joints[4+s].position={s*.2,.5,-.08};b.joints[6+s].position={s*.2,.1,0};}
 auto original=b;VirtualLegLift patch;
 for(int i=0;i<100;i++){b.time_ns+=14000000;auto q=patch.Apply(b);check(patch.swingSide<0&&patch.lift[0]==0&&patch.lift[1]==0,"Standing modified");}
 double maximumLift=0;
 for(int i=0;i<100;i++){
  b=original;b.time_ns=2400000000LL+(i+1)*14000000LL;
  b.joints[6].position[1]+=.065*std::sin(3.141592653589793*std::min(i,70)/70.);
  b.joints[6].position[2]-=.002*i;
  auto q=patch.Apply(b);auto repeat=patch.Apply(b);maximumLift=std::max(maximumLift,patch.lift[0]);
  for(int j=0;j<14;j++)check(q.joints[j].position==repeat.joints[j].position,"Duplicate frame changed patch");
  for(auto j:{0,1,3,5,7,8,9,10,11,12,13})check(q.joints[j].position==b.joints[j].position,"Unselected joint moved");
  check(std::abs(distance(q.joints[2],q.joints[4])-distance(b.joints[2],b.joints[4]))<1e-8,"Thigh stretched");
  check(std::abs(distance(q.joints[4],q.joints[6])-distance(b.joints[4],b.joints[6]))<1e-8,"Shin stretched");
  check(q.joints[6].position[0]==b.joints[6].position[0]&&q.joints[6].position[2]==b.joints[6].position[2],"Ankle XY changed");
 }
 check(maximumLift>.02,"Swing foot never lifted");
 b.time_ns+=1000000000;auto q=patch.Apply(b);check(patch.swingSide<0,"Gap retained swing phase");
 b.valid=false;patch.Apply(b);check(patch.swingSide<0,"Invalid input retained swing");
 std::cout<<"Virtual leg lift: stance, bone lengths, upper/support invariance, ankle XY, duplicates, gaps passed\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

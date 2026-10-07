#pragma once
#include <chrono>
// One toggle per press. An inactive controller is not a confirmed release.
class ButtonLatch {
 bool armed=true;double releasedAt=-1;
public:
 bool Update(bool active,bool pressed,double now=std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()){
  if(!active){releasedAt=-1;return false;}
  if(pressed){releasedAt=-1;if(!armed)return false;armed=false;return true;}
  if(releasedAt<0)releasedAt=now;
  if(now-releasedAt>=.1)armed=true;
  return false;
 }
};

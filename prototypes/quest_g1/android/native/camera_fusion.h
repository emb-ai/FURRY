#pragma once
#include "retarget.h"
#include <deque>
#include <string>
struct CameraJoint {std::array<double,3> p{};double confidence=0;bool measured=false;};
struct CameraSkeleton {
 // hips L/R, knees L/R, ankles L/R, nose, shoulders L/R, wrists L/R
 std::array<CameraJoint,11> joints{};CameraJoint pelvis;
 uint64_t sequence=0;double sourceMs=0,receivedMs=0;std::string frame;
};
struct CameraFusionStats {int state=0,legs=0;double ageMs=-1,fitMm=-1,weight=0;};
class CameraFusion {
 struct Sample{double ms;TrackingFrame frame;};
 std::deque<Sample> history;
 std::vector<std::array<double,3>> fitFrom,fitTo,headFrom,headTo;
 std::array<double,4> fitHeadRotation{1,0,0,0};
 std::array<double,9> rotation{1,0,0,0,1,0,0,0,1};
 std::array<double,3> translation{};
 std::array<std::array<double,3>,6> correction{};
 std::array<double,2> weights{},activeWeights{},lastGoodMs{};
 bool aligned=false;uint64_t lastSequence=~0ull;double lastPacketMs=0,lastApplyMs=0;
 void Fit(const CameraSkeleton&,const TrackingFrame&);
public:
 CameraFusionStats stats;
 void Reset();
 void Observe(const TrackingFrame&,double epochMs);
 TrackingFrame Apply(const TrackingFrame&,const CameraSkeleton*,double epochMs);
 bool Aligned()const{return aligned;}
};

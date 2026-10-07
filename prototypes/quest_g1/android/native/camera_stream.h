#pragma once
#include "camera_fusion.h"
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
namespace ix{class WebSocket;}
double CameraEpochMs();
class CameraStream {
 std::unique_ptr<ix::WebSocket> socket;
 mutable std::mutex mutex;
 CameraSkeleton latest;bool have=false;
 std::string pendingUplink,latestRaw;
 std::atomic<bool> running{false};std::thread sender;
public:
 std::atomic<bool> connected{false};std::atomic<uint64_t> rejected{0};
 explicit CameraStream(const std::string&url);
 ~CameraStream();
 void Submit(const TrackingFrame&);
 bool Latest(CameraSkeleton&,std::string* raw=nullptr)const;
 static bool Parse(const std::string&,double receivedMs,CameraSkeleton&);
};

#include "body_tracking.h"
#include <cassert>
#include <iostream>
static bool active=true,fail=false,malformed=false;static uint32_t version=1;static uint64_t ballFlags=15;
extern "C" {
XrResult xrEnumerateInstanceExtensionProperties(const char*,uint32_t,uint32_t*,XrExtensionProperties*){return XR_ERROR_RUNTIME_FAILURE;}
XrResult xrGetSystemProperties(XrInstance,XrSystemId,XrSystemProperties*p){auto*b=(XrSystemBodyTrackingPropertiesFB*)p->next;b->supportsBodyTracking=true;((XrSystemPropertiesBodyTrackingFullBodyMETA*)b->next)->supportsFullBodyTracking=true;return XR_SUCCESS;}
static XrResult Create(XrSession,const XrBodyTrackerCreateInfoFB*,XrBodyTrackerFB*t){*t=(XrBodyTrackerFB)1;return XR_SUCCESS;}
static XrResult Destroy(XrBodyTrackerFB){return XR_SUCCESS;}
static XrResult Locate(XrBodyTrackerFB,const XrBodyJointsLocateInfoFB*,XrBodyJointLocationsFB*l){
 if(fail)return XR_ERROR_RUNTIME_FAILURE;l->isActive=active;l->confidence=.9f;l->time=12345;l->skeletonChangedCount=version;
 for(int j=0;j<84;j++){l->jointLocations[j].locationFlags=j==83?ballFlags:15;l->jointLocations[j].pose={{0,0,0,1},{float(j)/100,1,2}};}
 return XR_SUCCESS;
}
static XrResult Skeleton(XrBodyTrackerFB,XrBodySkeletonFB*s){
 for(int i=0;i<84;i++){int j=83-i;s->joints[i].joint=(XrBodyJointFB)j;s->joints[i].parentJoint=(XrBodyJointFB)(j-1);s->joints[i].pose={{0,0,0,1},{float(j)/50,1,2}};}
 if(malformed)s->joints[1].joint=s->joints[0].joint;return XR_SUCCESS;
}
XrResult xrGetInstanceProcAddr(XrInstance,const char*n,PFN_xrVoidFunction*p){
 if(!strcmp(n,"xrCreateBodyTrackerFB"))*p=(PFN_xrVoidFunction)Create;
 else if(!strcmp(n,"xrDestroyBodyTrackerFB"))*p=(PFN_xrVoidFunction)Destroy;
 else if(!strcmp(n,"xrLocateBodyJointsFB"))*p=(PFN_xrVoidFunction)Locate;
 else if(!strcmp(n,"xrGetBodySkeletonFB"))*p=(PFN_xrVoidFunction)Skeleton;
 return XR_SUCCESS;
}
}
int main(){
 BodyTracking tracker;tracker.enabled=true;tracker.Initialize((XrInstance)1,1,(XrSession)1);TrackingFrame f;
 tracker.Sample((XrSpace)1,12345,f);assert(f.body.valid&&f.body.full&&f.body.full->skeleton);
 assert(f.body.full->joints[83].position[0]==double(float(83)/100));assert(f.body.full->skeleton->rest[76].position[0]==double(float(76)/50));assert(f.body.full->skeleton->parent[76]==75);
 auto immutable=f.body.full;ballFlags=0;tracker.Sample((XrSpace)1,12345,f);assert(f.body.valid&&f.body.full->flags[83]==0&&immutable->flags[83]==15);
 active=false;tracker.Sample((XrSpace)1,12345,f);assert(!f.body.valid&&f.body.full&&!f.body.full->active);
 active=true;malformed=true;version++;tracker.Sample((XrSpace)1,12345,f);assert(f.body.valid&&f.body.full&&!f.body.full->skeleton);
 fail=true;tracker.Sample((XrSpace)1,12345,f);assert(!f.body.full);
 tracker.Shutdown();std::cout<<"84-joint SDK capture, shuffled bind IDs, immutable samples, invalid extras and inactive/failing tracking passed\n";
}

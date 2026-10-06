#pragma once
#include "retarget.h"
#include <openxr/openxr.h>
#include <android/log.h>
#include <algorithm>
#include <cstring>
#include <cmath>

// Official OpenXR body stream. No poses are synthesized in this adapter.
class BodyTracking {
    XrBodyTrackerFB tracker=XR_NULL_HANDLE;
    PFN_xrDestroyBodyTrackerFB destroy=nullptr;
    PFN_xrLocateBodyJointsFB locate=nullptr;
    PFN_xrGetBodySkeletonFB skeleton=nullptr;
    std::array<XrBodyJointLocationFB,84> joints{};
    std::array<XrBodySkeletonJointFB,84> bind{};
    std::array<TrackedPose,14> bindSelected{};
    uint32_t version=~0u;
    bool haveSkeleton=false;
    static constexpr int indices[14]={1,5,70,77,71,78,73,80,10,15,11,16,19,45};
    static TrackedPose Pose(const XrPosef&p){return {{p.position.x,p.position.y,p.position.z},{p.orientation.w,p.orientation.x,p.orientation.y,p.orientation.z}};}
public:
    bool enabled=false;
    static bool Enable(std::vector<const char*>& extensions){
        uint32_t n=0;if(XR_FAILED(xrEnumerateInstanceExtensionProperties(nullptr,0,&n,nullptr)))return false;
        std::vector<XrExtensionProperties> props(n,{XR_TYPE_EXTENSION_PROPERTIES});
        if(XR_FAILED(xrEnumerateInstanceExtensionProperties(nullptr,n,&n,props.data())))return false;
        auto has=[&](const char* name){return std::any_of(props.begin(),props.end(),[&](const auto&p){return std::strcmp(p.extensionName,name)==0;});};
        if(!has(XR_FB_BODY_TRACKING_EXTENSION_NAME)||!has(XR_META_BODY_TRACKING_FULL_BODY_EXTENSION_NAME))return false;
        extensions.push_back(XR_FB_BODY_TRACKING_EXTENSION_NAME);extensions.push_back(XR_META_BODY_TRACKING_FULL_BODY_EXTENSION_NAME);return true;
    }
    void Initialize(XrInstance instance,XrSystemId system,XrSession session){
        if(!enabled){__android_log_print(ANDROID_LOG_WARN,"G1Quest","Full-body extensions unavailable: GMR input disabled");return;}
        XrSystemPropertiesBodyTrackingFullBodyMETA full{XR_TYPE_SYSTEM_PROPERTIES_BODY_TRACKING_FULL_BODY_META};
        XrSystemBodyTrackingPropertiesFB body{XR_TYPE_SYSTEM_BODY_TRACKING_PROPERTIES_FB,&full};
        XrSystemProperties properties{XR_TYPE_SYSTEM_PROPERTIES,&body};
        if(XR_FAILED(xrGetSystemProperties(instance,system,&properties))||!body.supportsBodyTracking||!full.supportsFullBodyTracking){enabled=false;return;}
        PFN_xrCreateBodyTrackerFB create=nullptr;
        xrGetInstanceProcAddr(instance,"xrCreateBodyTrackerFB",reinterpret_cast<PFN_xrVoidFunction*>(&create));
        xrGetInstanceProcAddr(instance,"xrDestroyBodyTrackerFB",reinterpret_cast<PFN_xrVoidFunction*>(&destroy));
        xrGetInstanceProcAddr(instance,"xrLocateBodyJointsFB",reinterpret_cast<PFN_xrVoidFunction*>(&locate));
        xrGetInstanceProcAddr(instance,"xrGetBodySkeletonFB",reinterpret_cast<PFN_xrVoidFunction*>(&skeleton));
        if(!create||!destroy||!locate||!skeleton){enabled=false;return;}
        XrBodyTrackerCreateInfoFB info{XR_TYPE_BODY_TRACKER_CREATE_INFO_FB};info.bodyJointSet=XR_BODY_JOINT_SET_FULL_BODY_META;
        auto result=create(session,&info,&tracker);
        __android_log_print(ANDROID_LOG_INFO,"G1Quest","Full body tracker create: %d",int(result));
        if(XR_FAILED(result)){tracker=XR_NULL_HANDLE;enabled=false;}
    }
    void Sample(XrSpace space,XrTime time,TrackingFrame& input){
        input.body.supported=enabled;
        if(!tracker)return;
        XrBodyJointsLocateInfoFB info{XR_TYPE_BODY_JOINTS_LOCATE_INFO_FB};info.baseSpace=space;info.time=time;
        XrBodyJointLocationsFB locations{XR_TYPE_BODY_JOINT_LOCATIONS_FB};locations.jointCount=84;locations.jointLocations=joints.data();
        if(XR_FAILED(locate(tracker,&info,&locations)))return;
        if(version!=locations.skeletonChangedCount){
            XrBodySkeletonFB desc{XR_TYPE_BODY_SKELETON_FB};desc.jointCount=84;desc.joints=bind.data();
            haveSkeleton=XR_SUCCEEDED(skeleton(tracker,&desc));
            if(haveSkeleton){
                for(int i=0;i<14;i++){
                    auto found=std::find_if(bind.begin(),bind.end(),[&](const auto&j){return j.joint==indices[i];});
                    if(found==bind.end()){haveSkeleton=false;break;}
                    bindSelected[i]=Pose(found->pose);
                }
                if(haveSkeleton)version=locations.skeletonChangedCount;
            }
        }
        auto& out=input.body;out.confidence=locations.confidence;out.time_ns=locations.time;out.skeleton_version=version;
        out.valid=locations.isActive && haveSkeleton && locations.confidence>0 && locations.time>0;
        for(int i=0;i<14;i++){
            out.joints[i]=Pose(joints[indices[i]].pose);out.flags[i]=joints[indices[i]].locationFlags;out.rest[i]=bindSelected[i];
            out.valid=out.valid && (out.flags[i]&3)==3;
            for(const auto&p:{out.joints[i],out.rest[i]}){
                double norm=0;for(double v:p.position)out.valid=out.valid && std::isfinite(v);
                for(double v:p.quaternion){out.valid=out.valid && std::isfinite(v);norm+=v*v;}
                out.valid=out.valid && std::abs(norm-1)<.01;
            }
        }
    }
    void Shutdown(){if(tracker && destroy)destroy(tracker);tracker=XR_NULL_HANDLE;}
};

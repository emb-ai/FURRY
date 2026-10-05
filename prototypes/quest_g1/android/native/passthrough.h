#pragma once
#include <openxr/openxr.h>
#include <android/log.h>
#include <stdexcept>
#include <string>

class Passthrough {
    XrPassthroughFB handle=XR_NULL_HANDLE;
    XrPassthroughLayerFB layer=XR_NULL_HANDLE;
    PFN_xrDestroyPassthroughFB destroy=nullptr;
    PFN_xrDestroyPassthroughLayerFB destroyLayer=nullptr;
    XrCompositionLayerPassthroughFB composition{XR_TYPE_COMPOSITION_LAYER_PASSTHROUGH_FB};
    static void Check(XrResult result,const char* what){
        if(XR_FAILED(result))throw std::runtime_error(std::string(what)+": "+std::to_string(result));
    }
    template<class T> static T Function(XrInstance instance,const char* name){
        T result=nullptr;
        Check(xrGetInstanceProcAddr(instance,name,reinterpret_cast<PFN_xrVoidFunction*>(&result)),name);
        return result;
    }
public:
    void Initialize(XrInstance instance,XrSession session){
        auto create=Function<PFN_xrCreatePassthroughFB>(instance,"xrCreatePassthroughFB");
        auto createLayer=Function<PFN_xrCreatePassthroughLayerFB>(instance,"xrCreatePassthroughLayerFB");
        destroy=Function<PFN_xrDestroyPassthroughFB>(instance,"xrDestroyPassthroughFB");
        destroyLayer=Function<PFN_xrDestroyPassthroughLayerFB>(instance,"xrDestroyPassthroughLayerFB");
        XrPassthroughCreateInfoFB info{XR_TYPE_PASSTHROUGH_CREATE_INFO_FB};
        info.flags=XR_PASSTHROUGH_IS_RUNNING_AT_CREATION_BIT_FB;
        Check(create(session,&info,&handle),"Create passthrough");
        XrPassthroughLayerCreateInfoFB layerInfo{XR_TYPE_PASSTHROUGH_LAYER_CREATE_INFO_FB};
        layerInfo.passthrough=handle;layerInfo.flags=XR_PASSTHROUGH_IS_RUNNING_AT_CREATION_BIT_FB;
        layerInfo.purpose=XR_PASSTHROUGH_LAYER_PURPOSE_RECONSTRUCTION_FB;
        Check(createLayer(session,&layerInfo,&layer),"Create passthrough layer");
        composition.layerHandle=layer;
        composition.space=XR_NULL_HANDLE;
        __android_log_print(ANDROID_LOG_INFO,"G1Quest","Color passthrough layer initialized");
    }
    void Shutdown(){
        if(layer!=XR_NULL_HANDLE){destroyLayer(layer);layer=XR_NULL_HANDLE;}
        if(handle!=XR_NULL_HANDLE){destroy(handle);handle=XR_NULL_HANDLE;}
    }
    XrCompositionLayerBaseHeader* Layer(){return reinterpret_cast<XrCompositionLayerBaseHeader*>(&composition);}
};

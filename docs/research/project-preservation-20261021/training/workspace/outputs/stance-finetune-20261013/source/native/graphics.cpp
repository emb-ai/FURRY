// OpenXR swapchain handling follows Khronos hello_xr (Apache-2.0).
#include "pch.h"
#include "common.h"
#include "graphicsplugin.h"
#include "options.h"
#include "quest_runtime.h"
#include <common/xr_linear.h>

namespace {
class Graphics : public IGraphicsPlugin {
    EGLDisplay display=EGL_NO_DISPLAY;
    EGLContext context=EGL_NO_CONTEXT;
    EGLSurface surface=EGL_NO_SURFACE;
    XrGraphicsBindingOpenGLESAndroidKHR binding{XR_TYPE_GRAPHICS_BINDING_OPENGL_ES_ANDROID_KHR};
    std::list<std::vector<XrSwapchainImageOpenGLESKHR>> images;
    std::map<GLuint,GLuint> depths;
    GLuint framebuffer=0;
public:
    ~Graphics() override {
        if(context!=EGL_NO_CONTEXT){
            for(auto [color,depth]:depths)glDeleteTextures(1,&depth);
            if(framebuffer)glDeleteFramebuffers(1,&framebuffer);
            eglMakeCurrent(display,EGL_NO_SURFACE,EGL_NO_SURFACE,EGL_NO_CONTEXT);
            eglDestroyContext(display,context);eglDestroySurface(display,surface);eglTerminate(display);
        }
    }
    std::vector<std::string> GetInstanceExtensions()const override{return {XR_KHR_OPENGL_ES_ENABLE_EXTENSION_NAME};}
    void InitializeDevice(XrInstance instance,XrSystemId system) override {
        PFN_xrGetOpenGLESGraphicsRequirementsKHR getRequirements=nullptr;
        CHECK_XRCMD(xrGetInstanceProcAddr(instance,"xrGetOpenGLESGraphicsRequirementsKHR",reinterpret_cast<PFN_xrVoidFunction*>(&getRequirements)));
        XrGraphicsRequirementsOpenGLESKHR requirements{XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_ES_KHR};
        CHECK_XRCMD(getRequirements(instance,system,&requirements));
        display=eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if(!eglInitialize(display,nullptr,nullptr))throw std::runtime_error("eglInitialize failed");
        const EGLint attributes[]={EGL_RENDERABLE_TYPE,EGL_OPENGL_ES3_BIT,EGL_SURFACE_TYPE,EGL_PBUFFER_BIT,EGL_RED_SIZE,8,EGL_GREEN_SIZE,8,EGL_BLUE_SIZE,8,EGL_ALPHA_SIZE,8,EGL_NONE};
        EGLConfig config;EGLint count;
        if(!eglChooseConfig(display,attributes,&config,1,&count)||count<1)throw std::runtime_error("No EGL config");
        const EGLint contextAttributes[]={EGL_CONTEXT_CLIENT_VERSION,3,EGL_NONE};
        context=eglCreateContext(display,config,EGL_NO_CONTEXT,contextAttributes);
        const EGLint surfaceAttributes[]={EGL_WIDTH,16,EGL_HEIGHT,16,EGL_NONE};
        surface=eglCreatePbufferSurface(display,config,surfaceAttributes);
        if(context==EGL_NO_CONTEXT||surface==EGL_NO_SURFACE||!eglMakeCurrent(display,surface,surface,context))throw std::runtime_error("EGL context failed");
        binding.display=display;binding.config=config;binding.context=context;
        glGenFramebuffers(1,&framebuffer);
    }
    int64_t SelectColorSwapchainFormat(const std::vector<int64_t>& formats)const override{
        for(auto preferred:{GL_SRGB8_ALPHA8,GL_RGBA8})if(std::find(formats.begin(),formats.end(),preferred)!=formats.end())return preferred;
        throw std::runtime_error("No supported swapchain format");
    }
    const XrBaseInStructure* GetGraphicsBinding()const override{return reinterpret_cast<const XrBaseInStructure*>(&binding);}
    std::vector<XrSwapchainImageBaseHeader*> AllocateSwapchainImageStructs(uint32_t count,const XrSwapchainCreateInfo&)override{
        images.emplace_back(count,XrSwapchainImageOpenGLESKHR{XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_ES_KHR});
        std::vector<XrSwapchainImageBaseHeader*> out;for(auto& image:images.back())out.push_back(reinterpret_cast<XrSwapchainImageBaseHeader*>(&image));return out;
    }
    void RenderView(const XrCompositionLayerProjectionView& view,const XrSwapchainImageBaseHeader* image,int64_t,const std::vector<Cube>&)override{
        GLuint color=reinterpret_cast<const XrSwapchainImageOpenGLESKHR*>(image)->image;
        auto rect=view.subImage.imageRect;
        auto it=depths.find(color);
        if(it==depths.end()){
            GLuint depth;glGenTextures(1,&depth);glBindTexture(GL_TEXTURE_2D,depth);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
            glTexImage2D(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT24,rect.extent.width,rect.extent.height,0,GL_DEPTH_COMPONENT,GL_UNSIGNED_INT,nullptr);
            it=depths.emplace(color,depth).first;
        }
        glBindFramebuffer(GL_FRAMEBUFFER,framebuffer);
        glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,color,0);
        glFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_TEXTURE_2D,it->second,0);
        if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)throw std::runtime_error("Incomplete XR framebuffer");
        glViewport(rect.offset.x,rect.offset.y,rect.extent.width,rect.extent.height);
        if(G1PassthroughVisible())glClearColor(0,0,0,0);else glClearColor(.06f,.07f,.06f,1);
        glClearDepthf(1);glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        XrMatrix4x4f projection,toView,viewMatrix,vp;
        XrMatrix4x4f_CreateProjectionFov(&projection,GRAPHICS_OPENGL_ES,view.fov,.04f,50.f);
        XrMatrix4x4f_CreateFromRigidTransform(&toView,&view.pose);
        XrMatrix4x4f_InvertRigidBody(&viewMatrix,&toView);
        XrMatrix4x4f_Multiply(&vp,&projection,&viewMatrix);
        G1Render(vp.m,projection.m);
        G1CaptureFrame(rect.extent.width,rect.extent.height);
        glBindFramebuffer(GL_FRAMEBUFFER,0);
    }
    uint32_t GetSupportedSwapchainSampleCount(const XrViewConfigurationView&)override{return 1;}
    void UpdateOptions(const std::shared_ptr<Options>&)override{}
};
}
std::shared_ptr<IGraphicsPlugin> CreateGraphicsPlugin_OpenGLES(const std::shared_ptr<Options>&,std::shared_ptr<IPlatformPlugin>){return std::make_shared<Graphics>();}

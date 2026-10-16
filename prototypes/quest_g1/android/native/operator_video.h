#pragma once
#include <jni.h>
#include <string>
#include <stdexcept>

// Java owns capture on the Android main thread; no GL readback or encoder work
// happens on the physics/render threads. Method IDs are immutable after init.
class OperatorVideo {
    struct Env {
        JavaVM* vm; JNIEnv* env=nullptr; bool attached=false;
        explicit Env(JavaVM* v):vm(v) {
            if(vm->GetEnv(reinterpret_cast<void**>(&env),JNI_VERSION_1_6)==JNI_EDETACHED){
                if(vm->AttachCurrentThread(&env,nullptr)!=JNI_OK)throw std::runtime_error("Video JNI attach failed");
                attached=true;
            }
            if(!env)throw std::runtime_error("Video JNI unavailable");
        }
        ~Env(){if(attached)vm->DetachCurrentThread();}
        void Check(){if(env->ExceptionCheck()){env->ExceptionDescribe();env->ExceptionClear();throw std::runtime_error("Video JNI call failed");}}
    };
    JavaVM* vm_; jobject activity_=nullptr;
    jmethodID toggle_,stop_,state_,message_;
public:
    int state=0;
    std::string message="Видео недоступно";
    OperatorVideo(JavaVM* vm,jobject activity):vm_(vm){
        Env e(vm_);jclass cls=e.env->GetObjectClass(activity);e.Check();
        toggle_=e.env->GetMethodID(cls,"toggleOperatorVideo","()V");
        stop_=e.env->GetMethodID(cls,"stopOperatorVideo","()V");
        state_=e.env->GetMethodID(cls,"getOperatorVideoState","()I");
        message_=e.env->GetMethodID(cls,"getOperatorVideoMessage","()Ljava/lang/String;");
        e.env->DeleteLocalRef(cls);e.Check();
        activity_=e.env->NewGlobalRef(activity);e.Check();
    }
    ~OperatorVideo(){try{Env e(vm_);e.env->DeleteGlobalRef(activity_);}catch(...) {}}
    void Toggle(){Env e(vm_);e.env->CallVoidMethod(activity_,toggle_);e.Check();}
    void Stop(){Env e(vm_);e.env->CallVoidMethod(activity_,stop_);e.Check();}
    void Poll(){
        Env e(vm_);state=e.env->CallIntMethod(activity_,state_);e.Check();
        auto str=static_cast<jstring>(e.env->CallObjectMethod(activity_,message_));e.Check();
        if(str){const char* text=e.env->GetStringUTFChars(str,nullptr);
            if(text){message=text;e.env->ReleaseStringUTFChars(str,text);}
            e.env->DeleteLocalRef(str);e.Check();}
    }
};

#pragma once
#include <jni.h>
#include <android/log.h>
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <queue>
#include <thread>
#include <stdexcept>

// Publish a local ZIP through MediaStore Downloads. No storage permission or
// network is needed; JNI and compression run off the physics/render threads.
class RecordingExport {
    struct Env {
        JavaVM* vm; JNIEnv* e=nullptr; bool attached=false;
        explicit Env(JavaVM* v):vm(v){
            int result=vm->GetEnv(reinterpret_cast<void**>(&e),JNI_VERSION_1_6);
            if(result==JNI_EDETACHED){
#ifdef __ANDROID__
                int attach=vm->AttachCurrentThread(&e,nullptr);
#else
                int attach=vm->AttachCurrentThread(reinterpret_cast<void**>(&e),nullptr);
#endif
                if(attach!=JNI_OK)throw std::runtime_error("Export JNI attach failed");attached=true;
            }
            else if(result!=JNI_OK)throw std::runtime_error("Export JNI unavailable");
        }
        ~Env(){if(attached)vm->DetachCurrentThread();}
    };
    JavaVM* vm; jobject activity=nullptr; std::thread worker;
    std::mutex mutex; std::condition_variable ready;
    std::queue<std::filesystem::path> pending; bool stopping=false;
    static void Check(JNIEnv* e,const char* operation){
        if(e->ExceptionCheck()){e->ExceptionDescribe();e->ExceptionClear();throw std::runtime_error(operation);}
    }
    void Publish(const std::filesystem::path& directory){
        const auto marker=directory/".download-exported";
        if(std::filesystem::exists(marker))return;
        Env env(vm);auto* e=env.e;
        if(e->PushLocalFrame(128)<0){Check(e,"Export local frame failed");throw std::runtime_error("Export local frame failed");}
        struct Frame {JNIEnv* e;~Frame(){e->PopLocalFrame(nullptr);}} frame{e};
        auto cls=[&](const char* name){auto c=e->FindClass(name);Check(e,name);return c;};
        auto method=[&](jclass c,const char* name,const char* sig){auto m=e->GetMethodID(c,name,sig);Check(e,name);return m;};
        auto activityClass=cls("android/app/Activity"),resolverClass=cls("android/content/ContentResolver");
        jobject resolver=e->CallObjectMethod(activity,method(activityClass,"getContentResolver","()Landroid/content/ContentResolver;"));Check(e,"getContentResolver");
        auto downloads=cls("android/provider/MediaStore$Downloads");
        auto uriField=e->GetStaticFieldID(downloads,"EXTERNAL_CONTENT_URI","Landroid/net/Uri;");Check(e,"Downloads URI");
        jobject collection=e->GetStaticObjectField(downloads,uriField);Check(e,"Downloads URI value");
        auto valuesClass=cls("android/content/ContentValues");jobject values=e->NewObject(valuesClass,method(valuesClass,"<init>","()V"));Check(e,"ContentValues");
        auto putString=method(valuesClass,"put","(Ljava/lang/String;Ljava/lang/String;)V");
        auto put=[&](const char* key,const std::string& value){jstring k=e->NewStringUTF(key),v=e->NewStringUTF(value.c_str());Check(e,"Export metadata allocation");e->CallVoidMethod(values,putString,k,v);Check(e,"Export metadata");e->DeleteLocalRef(k);e->DeleteLocalRef(v);};
        auto integer=cls("java/lang/Integer");auto valueOf=e->GetStaticMethodID(integer,"valueOf","(I)Ljava/lang/Integer;");Check(e,"Integer.valueOf");
        auto putInteger=method(valuesClass,"put","(Ljava/lang/String;Ljava/lang/Integer;)V");
        auto setPending=[&](int value){jstring key=e->NewStringUTF("is_pending");jobject boxed=e->CallStaticObjectMethod(integer,valueOf,value);Check(e,"Pending value");e->CallVoidMethod(values,putInteger,key,boxed);Check(e,"Pending metadata");e->DeleteLocalRef(key);e->DeleteLocalRef(boxed);};
        std::string filename="g1-"+directory.filename().string()+".zip";
        put("_display_name",filename);put("mime_type","application/zip");put("relative_path","Download/G1Quest/");setPending(1);
        auto insert=method(resolverClass,"insert","(Landroid/net/Uri;Landroid/content/ContentValues;)Landroid/net/Uri;");
        auto remove=method(resolverClass,"delete","(Landroid/net/Uri;Ljava/lang/String;[Ljava/lang/String;)I");
        auto update=method(resolverClass,"update","(Landroid/net/Uri;Landroid/content/ContentValues;Ljava/lang/String;[Ljava/lang/String;)I");
        jobject uri=e->CallObjectMethod(resolver,insert,collection,values);Check(e,"Create Downloads entry");if(!uri)throw std::runtime_error("Downloads entry unavailable");
        jobject output=nullptr,zip=nullptr; jmethodID closeOutput=nullptr,closeZip=nullptr;
        try{
            output=e->CallObjectMethod(resolver,method(resolverClass,"openOutputStream","(Landroid/net/Uri;)Ljava/io/OutputStream;"),uri);Check(e,"Open Downloads stream");if(!output)throw std::runtime_error("Downloads stream unavailable");
            closeOutput=method(cls("java/io/OutputStream"),"close","()V");
            auto zipClass=cls("java/util/zip/ZipOutputStream"),entryClass=cls("java/util/zip/ZipEntry");
            zip=e->NewObject(zipClass,method(zipClass,"<init>","(Ljava/io/OutputStream;)V"),output);Check(e,"Create ZIP stream");
            closeZip=method(zipClass,"close","()V");
            e->CallVoidMethod(zip,method(zipClass,"setLevel","(I)V"),1);Check(e,"ZIP compression level");
            auto entryConstructor=method(entryClass,"<init>","(Ljava/lang/String;)V");
            auto beginEntry=method(zipClass,"putNextEntry","(Ljava/util/zip/ZipEntry;)V"),endEntry=method(zipClass,"closeEntry","()V");
            auto write=method(zipClass,"write","([BII)V");jbyteArray bytes=e->NewByteArray(65536);Check(e,"ZIP buffer");char buffer[65536];
            int count=0;
            for(const auto& file:std::filesystem::directory_iterator(directory)){
                if(!file.is_regular_file() || file.path().filename().string().front()=='.')continue;
                std::ifstream input(file.path(),std::ios::binary);if(!input)throw std::runtime_error("Cannot read episode file");
                std::string entryName=directory.filename().string()+"/"+file.path().filename().string();
                jstring name=e->NewStringUTF(entryName.c_str());jobject entry=e->NewObject(entryClass,entryConstructor,name);Check(e,"ZIP entry");
                e->CallVoidMethod(zip,beginEntry,entry);Check(e,"Begin ZIP entry");
                while(input){input.read(buffer,sizeof(buffer));auto n=input.gcount();if(n){e->SetByteArrayRegion(bytes,0,n,reinterpret_cast<jbyte*>(buffer));Check(e,"ZIP buffer copy");e->CallVoidMethod(zip,write,bytes,0,jint(n));Check(e,"Write ZIP entry");}}
                if(input.bad())throw std::runtime_error("Episode read failed");
                e->CallVoidMethod(zip,endEntry);Check(e,"Close ZIP entry");e->DeleteLocalRef(name);e->DeleteLocalRef(entry);count++;
            }
            if(!count)throw std::runtime_error("Empty episode");
            e->CallVoidMethod(zip,closeZip);Check(e,"Finalize ZIP");zip=nullptr;output=nullptr;
            e->CallVoidMethod(values,method(valuesClass,"clear","()V"));Check(e,"Clear export metadata");setPending(0);
            int updated=e->CallIntMethod(resolver,update,uri,values,nullptr,nullptr);Check(e,"Publish ZIP");if(updated!=1)throw std::runtime_error("ZIP publication failed");
            std::ofstream done(marker);done<<filename<<'\n';
            __android_log_print(ANDROID_LOG_INFO,"G1Quest","Recording exported: Download/G1Quest/%s",filename.c_str());
        }catch(...){
            if(zip && closeZip)e->CallVoidMethod(zip,closeZip);
            else if(output && closeOutput)e->CallVoidMethod(output,closeOutput);
            if(e->ExceptionCheck())e->ExceptionClear();
            e->CallIntMethod(resolver,remove,uri,nullptr,nullptr);if(e->ExceptionCheck())e->ExceptionClear();throw;
        }
    }
public:
    // 0 idle, 1 copying, 2 saved, 3 failed; originals are always retained.
    std::atomic<int> status{0};
    RecordingExport(JavaVM* v,jobject a):vm(v){Env env(vm);activity=env.e->NewGlobalRef(a);Check(env.e,"Activity global reference");if(!activity)throw std::runtime_error("Export activity unavailable");
        worker=std::thread([this]{for(;;){std::filesystem::path path;{std::unique_lock<std::mutex> lock(mutex);ready.wait(lock,[&]{return stopping||!pending.empty();});if(pending.empty())return;path=pending.front();pending.pop();}
            status=1;try{Publish(path);status=2;}catch(const std::exception& e){status=3;__android_log_print(ANDROID_LOG_ERROR,"G1Quest","Recording export failed: %s",e.what());}
        }});
    }
    ~RecordingExport(){Stop();try{Env env(vm);env.e->DeleteGlobalRef(activity);}catch(...){} }
    void Queue(const std::filesystem::path& path){std::lock_guard<std::mutex> lock(mutex);if(stopping)return;pending.push(path);ready.notify_one();}
    void Latest(const std::filesystem::path& root){
        std::filesystem::path latest;
        if(std::filesystem::exists(root))for(const auto& p:std::filesystem::directory_iterator(root))
            if(p.is_directory() && std::filesystem::is_regular_file(p.path()/"input.csv") && p.path().filename()>latest.filename())latest=p.path();
        if(!latest.empty())Queue(latest);
    }
    void Stop(){{std::lock_guard<std::mutex> lock(mutex);stopping=true;}ready.notify_one();if(worker.joinable())worker.join();}
};

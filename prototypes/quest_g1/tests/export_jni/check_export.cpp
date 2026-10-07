#include "recording_export.h"
#include <cassert>
int main(int argc,char**argv){
    if(argc!=4)return 2;
    std::string classpath="-Djava.class.path="+std::string(argv[1]),output="-Dexport.root="+std::string(argv[2]);
    JavaVMOption options[2]={{classpath.data(),nullptr},{output.data(),nullptr}};
    JavaVMInitArgs args{};args.version=JNI_VERSION_1_6;args.nOptions=2;args.options=options;
    JavaVM* vm=nullptr;JNIEnv* env=nullptr;
    assert(JNI_CreateJavaVM(&vm,reinterpret_cast<void**>(&env),&args)==JNI_OK);
    auto type=env->FindClass("android/app/Activity");auto activity=env->NewObject(type,env->GetMethodID(type,"<init>","()V"));
    auto root=std::filesystem::path(argv[3]);
    {
        RecordingExport exporter(vm,activity);exporter.Latest(root);exporter.Stop();assert(exporter.status==2);
        assert(std::filesystem::exists(root/"episode-002"/".download-exported"));
        assert(!std::filesystem::exists(root/"episode-001"/".download-exported"));
    }
    {
        RecordingExport exporter(vm,activity);exporter.Latest(root);exporter.Queue(root/"episode-001");exporter.Stop();assert(exporter.status==2);
    }
    std::filesystem::create_directory(root/"episode-fail");std::ofstream(root/"episode-fail"/"input.csv")<<"keep originals";
    {
        RecordingExport exporter(vm,activity);exporter.Queue(root/"episode-fail");exporter.Stop();assert(exporter.status==3);
        assert(!std::filesystem::exists(root/"episode-fail"/".download-exported"));
        assert(std::filesystem::exists(root/"episode-fail"/"input.csv"));
    }
    env->DeleteLocalRef(activity);env->DeleteLocalRef(type);vm->DestroyJavaVM();
}

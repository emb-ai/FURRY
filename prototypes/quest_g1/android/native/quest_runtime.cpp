#include "quest_runtime.h"
#include "simulation.h"
#include "meta_retarget.h"
#include "legend.h"
#include "recording.h"
#include "tracking_space.h"
#include <android_native_app_glue.h>
#include <android/asset_manager.h>
#include <android/log.h>
#include <GLES3/gl32.h>
#include <common/xr_linear.h>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <thread>

namespace {
std::unique_ptr<Simulation> sim;
std::unique_ptr<MetaRetargeter> retarget;
std::array<float,35> blendOrigin{};
double blendStarted=0;
bool wasApplying=false;
TrackingFrame latestTracking;
TrackingSpaceChanges spaceChanges;
EpisodeRecorder recorder;
std::atomic<int> recordingStatus{0};
std::atomic<bool> toggleRecording{false};
std::chrono::steady_clock::time_point trackingTime;
std::atomic<bool> calibrate{false},toggleTracking{false};
bool trackingEnabled=false;
bool faulted=false;
std::ofstream trace;
size_t traceRows=0;
std::array<float,29> lastReference{};
std::atomic<int> trackingStatus{0};
std::atomic<bool> placeScene{true};
std::array<float,16> sceneTransform{0,0,1,0, 1,0,0,0, 0,1,0,0, 0,0,-2.5,1};
std::mutex mutex;
std::thread worker;
std::atomic<bool> running{false}, active{false}, reset{false};
std::atomic<float> grip[2]{{0},{0}};
std::string assetsPath;
mjvScene scene{};
GLuint program=0;
struct Mesh {GLuint vao=0,vbo=0; GLsizei count=0;};
std::map<int,Mesh> meshes;
std::vector<std::vector<float>> visualMeshes;
GLint vpLocation, modelLocation, colorLocation;

template<class F> void Record(F work){
    try{work();recordingStatus=recorder.active?1:(recorder.failed?2:0);}
    catch(const std::exception& e){recorder.Abort();recordingStatus=2;__android_log_print(ANDROID_LOG_ERROR,"G1Quest","Recording failed: %s",e.what());}
}

void CopyAssets(AAssetManager* manager, const std::string& prefix, const std::filesystem::path& dest) {
    std::filesystem::create_directories(dest);
    AAssetDir* dir=AAssetManager_openDir(manager,prefix.c_str());
    if(!dir) throw std::runtime_error("Cannot open assets directory");
    while(const char* name=AAssetDir_getNextFileName(dir)) {
        std::string path=prefix.empty()?name:prefix+"/"+name;
        AAsset* asset=AAssetManager_open(manager,path.c_str(),AASSET_MODE_STREAMING);
        if(!asset) continue;
        std::ofstream output(dest/name,std::ios::binary|std::ios::trunc);
        if(!output) {AAsset_close(asset); throw std::runtime_error("Cannot write extracted asset");}
        char buffer[65536]; int count;
        while((count=AAsset_read(asset,buffer,sizeof(buffer)))>0) output.write(buffer,count);
        AAsset_close(asset);
    }
    AAssetDir_close(dir);
}

GLuint Shader(GLenum type,const char* source){
    GLuint shader=glCreateShader(type); glShaderSource(shader,1,&source,nullptr); glCompileShader(shader);
    GLint ok=0;glGetShaderiv(shader,GL_COMPILE_STATUS,&ok);
    if(!ok){char log[2048];glGetShaderInfoLog(shader,sizeof(log),nullptr,log);throw std::runtime_error(log);}
    return shader;
}
void InitGraphics(){
    if(program) return;
    const char* vs=R"(#version 320 es
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
uniform mat4 vp;
uniform mat4 model;
out vec3 worldNormal;
void main(){gl_Position=vp*model*vec4(position,1);worldNormal=normalize(mat3(model)*normal);}
)";
    const char* fs=R"(#version 320 es
precision mediump float;
in vec3 worldNormal;
uniform vec4 color;
out vec4 fragColor;
void main(){float light=.4+.6*max(dot(normalize(worldNormal),normalize(vec3(.5,1,.4))),0.);fragColor=vec4(color.rgb*light,color.a);}
)";
    GLuint v=Shader(GL_VERTEX_SHADER,vs),f=Shader(GL_FRAGMENT_SHADER,fs);
    program=glCreateProgram();glAttachShader(program,v);glAttachShader(program,f);glLinkProgram(program);
    GLint linked;glGetProgramiv(program,GL_LINK_STATUS,&linked);
    if(!linked) throw std::runtime_error("G1 shader link failed");
    glDeleteShader(v);glDeleteShader(f);
    vpLocation=glGetUniformLocation(program,"vp"); modelLocation=glGetUniformLocation(program,"model");colorLocation=glGetUniformLocation(program,"color");
}
void Triangle(std::vector<float>& out,const std::array<float,3>& a,const std::array<float,3>& b,const std::array<float,3>& c){
    float u[3]={b[0]-a[0],b[1]-a[1],b[2]-a[2]},v[3]={c[0]-a[0],c[1]-a[1],c[2]-a[2]};
    float n[3]={u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]};
    float length=std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
    for(auto p:{a,b,c}){for(float x:p)out.push_back(x);for(float x:n)out.push_back(length>1e-10?x/length:0);}
}
Mesh Upload(const std::vector<float>& verts){
    Mesh out; out.count=verts.size()/6;
    glGenVertexArrays(1,&out.vao);glBindVertexArray(out.vao);
    glGenBuffers(1,&out.vbo);glBindBuffer(GL_ARRAY_BUFFER,out.vbo);
    glBufferData(GL_ARRAY_BUFFER,verts.size()*sizeof(float),verts.data(),GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,6*sizeof(float),nullptr);
    glEnableVertexAttribArray(1);glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,6*sizeof(float),(void*)(3*sizeof(float)));
    return out;
}
Mesh& Geometry(const mjvGeom& g){
    // mjvGeom encodes mesh i as 2*i (or 2*i+1 for its convex hull).
    int key=g.type==mjGEOM_MESH ? g.dataid/2 : -1-g.type;
    auto it=meshes.find(key); if(it!=meshes.end())return it->second;
    std::vector<float> v;
    if(g.type==mjGEOM_MESH){
        if(key>=0 && key<static_cast<int>(visualMeshes.size()))
            return meshes.emplace(key,Upload(visualMeshes[key])).first->second;
        const mjModel* m=sim->model;int id=key;
        if(id<0 || id>=m->nmesh)throw std::runtime_error("Invalid scene mesh index");
        int start=m->mesh_faceadr[id],n=m->mesh_facenum[id],va=m->mesh_vertadr[id];
        for(int i=start;i<start+n;i++){
            std::array<float,3> p[3];
            for(int k=0;k<3;k++)for(int j=0;j<3;j++)p[k][j]=m->mesh_vert[3*(va+m->mesh_face[3*i+k])+j];
            Triangle(v,p[0],p[1],p[2]);
        }
    }else if(g.type==mjGEOM_BOX || g.type==mjGEOM_PLANE){
        std::array<float,3> p[8];for(int i=0;i<8;i++)p[i]={i&1?1.f:-1.f,i&2?1.f:-1.f,i&4?1.f:-1.f};
        int faces[][4]={{0,2,3,1},{4,5,7,6},{0,1,5,4},{2,6,7,3},{0,4,6,2},{1,3,7,5}};
        for(auto& f:faces){Triangle(v,p[f[0]],p[f[1]],p[f[2]]);Triangle(v,p[f[0]],p[f[2]],p[f[3]]);}
    }else{
        // Unit cylinder or sphere; capsules use a cylinder approximation only
        // for graphics. MuJoCo retains the exact physical capsule geometry.
        const int slices=16,rings=(g.type==mjGEOM_SPHERE || g.type==mjGEOM_ELLIPSOID)?8:1;
        bool sphere=rings>1;
        auto point=[&](int i,int j){float a=2*3.14159265f*i/slices;float z=sphere?-std::cos(3.14159265f*j/rings):float(j*2-1);float r=sphere?std::sin(3.14159265f*j/rings):1.f;return std::array<float,3>{r*std::cos(a),r*std::sin(a),z};};
        for(int j=0;j<rings;j++)for(int i=0;i<slices;i++){auto a=point(i,j),b=point(i+1,j),c=point(i+1,j+1),d=point(i,j+1);Triangle(v,a,b,c);Triangle(v,a,c,d);}
        if(!sphere)for(int i=0;i<slices;i++){Triangle(v,{0,0,-1},point(i+1,0),point(i,0));Triangle(v,{0,0,1},point(i,1),point(i+1,1));}
    }
    return meshes.emplace(key,Upload(v)).first->second;
}
}

void G1Initialize(android_app* app){
    std::string assets=std::string(app->activity->internalDataPath)+"/g1";
    assetsPath=assets;
    CopyAssets(app->activity->assetManager,"",assets);
    CopyAssets(app->activity->assetManager,"meshes",assets+"/meshes");
    sim=std::make_unique<Simulation>(assets);
    retarget=std::make_unique<MetaRetargeter>(assets);
    if(std::filesystem::exists(assets+"/teleop_trace.csv"))
        std::filesystem::rename(assets+"/teleop_trace.csv",assets+"/teleop_trace_previous.csv");
    trace.open(assets+"/teleop_trace.csv",std::ios::trunc);traceRows=0;
    trace<<"step,time,reset,apply_reference,grip_left,grip_right";
    for(int i=0;i<29;i++)trace<<",reference_"<<i;
    trace<<",tracking_valid,mode,ik_error,limited,height,root_qw,root_qx,root_qy,root_qz";
    for(auto name:{"head","left","right"})for(int i=0;i<7;i++)trace<<","<<name<<"_"<<i;
    trace<<"\n"<<std::setprecision(9);
    std::ifstream visual(assets+"/visual_meshes.bin",std::ios::binary);
    uint32_t count=0;visual.read(reinterpret_cast<char*>(&count),4);
    if(count!=sim->model->nmesh)throw std::runtime_error("Visual mesh count mismatch");
    visualMeshes.resize(count);
    for(auto& mesh:visualMeshes){uint32_t n=0;visual.read(reinterpret_cast<char*>(&n),4);if(n>3000000)throw std::runtime_error("Invalid visual mesh");mesh.resize(size_t(n)*6);visual.read(reinterpret_cast<char*>(mesh.data()),mesh.size()*sizeof(float));}
    if(!visual)throw std::runtime_error("Truncated visual mesh asset");
    mjv_defaultScene(&scene);mjv_makeScene(sim->model,&scene,2048);
    running=true;
    worker=std::thread([]{
        auto next=std::chrono::steady_clock::now();
        while(running){
            {
                std::lock_guard<std::mutex> guard(mutex);
                Record([]{
                    if(toggleRecording.exchange(false)){
                        if(recorder.active){recorder.Stop();__android_log_print(ANDROID_LOG_INFO,"G1Quest","Recording saved: %s",recorder.path().c_str());}
                        else{
                            recorder.Start(assetsPath,sim->model);
                            auto received=std::chrono::duration_cast<std::chrono::nanoseconds>(trackingTime.time_since_epoch()).count();
                            recorder.Input(latestTracking,grip[0],grip[1],received);
                            __android_log_print(ANDROID_LOG_INFO,"G1Quest","Recording started: %s",recorder.path().c_str());
                        }
                    }
                    recorder.Tick();
                });
            }
            if(!active || (faulted && !reset.load())){std::this_thread::sleep_for(std::chrono::milliseconds(10));next=std::chrono::steady_clock::now();continue;}
            try{
                std::lock_guard<std::mutex> guard(mutex);
                bool resetNow=reset.exchange(false);
                if(resetNow){sim->Reset();wasApplying=false;trackingEnabled=false;retarget->calibrated=false;trackingStatus=0;calibrate=false;toggleTracking=false;faulted=false;lastReference={};grip[0]=0;grip[1]=0;Record([]{recorder.Event("reset",latestTracking.sequence);});}
                bool valid=latestTracking.body.valid && (latestTracking.location_flags[0]&3)==3 && std::abs(latestTracking.xr_time_ns-latestTracking.body.time_ns)<200000000LL && std::chrono::steady_clock::now()-trackingTime<std::chrono::milliseconds(200);
                bool calibratedNow=false;
                if(calibrate.load() && valid){
                    calibratedNow=true;Record([]{recorder.Event("calibrate",latestTracking.sequence);});
                    retarget->Calibrate(sim->model,sim->data,latestTracking);trackingEnabled=true;calibrate=false;
                    __android_log_print(ANDROID_LOG_INFO,"G1Quest","Tracking calibrated: Meta full body -> native GMR -> TWIST2");
                }
                if(toggleTracking.exchange(false) && retarget->calibrated){trackingEnabled=!trackingEnabled;Record([]{recorder.Event(trackingEnabled?"tracking_resume":"tracking_pause",latestTracking.sequence);});}
                if(retarget->calibrated && valid && !retarget->Compatible(latestTracking)){
                    trackingEnabled=false;retarget->calibrated=false;
                    Record([]{recorder.Event("body_skeleton_changed",latestTracking.sequence);});
                }
                bool applyReference=trackingEnabled && valid && retarget->calibrated;
                float leftGrip=grip[0].load(),rightGrip=grip[1].load();
                if(applyReference){
                    if(calibratedNow || !wasApplying){
                        blendOrigin.fill(0);blendOrigin[2]=sim->data->qpos[2];blendStarted=sim->data->time;
                        auto* gm=retarget->solver().model();
                        for(int j=1;j<gm->njnt;j++){
                            int source=mj_name2id(sim->model,mjOBJ_JOINT,mj_id2name(gm,mjOBJ_JOINT,j));
                            blendOrigin[6+gm->jnt_qposadr[j]-7]=sim->data->qpos[sim->model->jnt_qposadr[source]];
                        }
                    }
                    auto whole=retarget->Solve(latestTracking);
                    double u=std::clamp((sim->data->time-blendStarted)/.5,0.,1.);
                    for(int j=0;j<35;j++)whole[j]=blendOrigin[j]+u*(whole[j]-blendOrigin[j]);
                    sim->SetWholeBodyReference(whole);std::copy_n(whole.begin()+6,29,lastReference.begin());
                }else{sim->PauseWholeBodyReference();retarget->Pause();}
                wasApplying=applyReference;
                trackingStatus=!valid?3:retarget->calibrated?(trackingEnabled?(valid?(retarget->error>.3?5:1):3):2):0;
                Record([&]{recorder.Frame(sim->model,sim->data,sim->steps,latestTracking,resetNow,calibratedNow,applyReference,valid,
                    leftGrip,rightGrip,trackingStatus,retarget->error,(retarget->error>.3),lastReference);});
                Record([&]{recorder.Mimic(latestTracking,sim->steps,sim->data->time,sim->WholeBodyReference());});
                // Keep the actual input episode for deterministic policy replay.
                if(trace && traceRows++<60000){
                    trace<<sim->steps<<","<<sim->data->time<<","<<resetNow<<","<<applyReference<<","<<leftGrip<<","<<rightGrip;
                    for(auto q:lastReference)trace<<","<<q;
                    trace<<","<<valid<<","<<trackingStatus.load()<<","<<retarget->error<<","<<(retarget->error>.3)<<","<<sim->data->qpos[2];
                    for(int i=3;i<7;i++)trace<<","<<sim->data->qpos[i];
                    for(const auto& pose:{latestTracking.head,latestTracking.hands[0],latestTracking.hands[1]}){
                        for(auto p:pose.position)trace<<","<<p;for(auto q:pose.quaternion)trace<<","<<q;
                    }
                    trace<<"\n";if(sim->steps%1000==0)trace.flush();
                }
                for(int i=0;i<10;i++)sim->Step(false,leftGrip,rightGrip);
                if(sim->steps%5000==0)__android_log_print(ANDROID_LOG_INFO,"G1Quest","sim=%.2fs height=%.3f inference=%.3fms",sim->data->time,sim->data->qpos[2],sim->inference_ms);
                if(sim->steps%5000==0)__android_log_print(ANDROID_LOG_INFO,"G1Quest","tracking state=%d GMR task residual=%.3f",trackingStatus.load(),retarget->error);
            }catch(const std::exception& e){
                __android_log_print(ANDROID_LOG_ERROR,"G1Quest","Simulation paused at %.3fs: %s; trace saved",sim->data->time,e.what());
                std::lock_guard<std::mutex> guard(mutex);trace.flush();Record([]{recorder.Event("simulation_fault",latestTracking.sequence);recorder.Flush();});faulted=true;trackingEnabled=false;calibrate=false;trackingStatus=4;
            }
            next+=std::chrono::milliseconds(10);std::this_thread::sleep_until(next);
            if(std::chrono::steady_clock::now()-next>std::chrono::milliseconds(100))next=std::chrono::steady_clock::now();
        }
    });
    __android_log_print(ANDROID_LOG_INFO,"G1Quest","Initialized autonomous MuJoCo %s, %d actuators",mj_versionString(),sim->model->nu);
}
void G1Shutdown(bool interrupted){
    running=false;if(worker.joinable())worker.join();
    Record([&]{
        if(interrupted){recorder.Event("app_error",latestTracking.sequence);recorder.Flush();recorder.Abort();}
        else recorder.Stop("app_shutdown");
    });
    trace.close();mjv_freeScene(&scene);retarget.reset();sim.reset();
}
void G1SetActive(bool value){
    if(active.exchange(value)!=value){
        std::lock_guard<std::mutex> guard(mutex);
        if(!value){wasApplying=false;if(retarget)retarget->Pause();if(sim)sim->PauseWholeBodyReference();}
        Record([&]{recorder.Event(value?"focus_resume":"focus_pause",latestTracking.sequence);});
    }
}
void G1Grip(int hand,float value){if(hand>=0 && hand<2)grip[hand]=value;}
void G1Reset(){reset=true;placeScene=true;}
void G1ReferenceSpaceChange(int64_t changeTime){
    std::lock_guard<std::mutex> guard(mutex);
    spaceChanges.Schedule(changeTime);
}
void G1SubmitTracking(const TrackingFrame& input){
    std::lock_guard<std::mutex> guard(mutex);latestTracking=input;trackingTime=std::chrono::steady_clock::now();
    if(spaceChanges.Advance(input.xr_time_ns)){
        trackingEnabled=false;retarget->calibrated=false;calibrate=false;toggleTracking=false;
        trackingStatus=faulted?4:0;placeScene=true;
        Record([&]{recorder.Event("reference_space_change",input.sequence);});
    }
    Record([&]{recorder.Input(input,grip[0],grip[1],EpisodeRecorder::Now());});
    if((input.location_flags[0]&3)==3 && placeScene.exchange(false)){
        double rotation[9];mju_quat2Mat(rotation,input.head.quaternion.data());
        double fx=-rotation[2],fz=-rotation[8],norm=std::hypot(fx,fz);
        if(norm<.2){fx=0;fz=-1;norm=1;}fx/=norm;fz/=norm;
        sceneTransform={float(-fx),0,float(-fz),0, float(-fz),0,float(fx),0, 0,1,0,0,
                        float(input.head.position[0]+2.1*fx),0,float(input.head.position[2]+2.1*fz),1};
        __android_log_print(ANDROID_LOG_INFO,"G1Quest","Scene placed 2.1m ahead of headset on stage floor");
    }
}
void G1Calibrate(){calibrate=true;}
void G1ToggleRecording(){toggleRecording=true;}
void G1ToggleTracking(){toggleTracking=true;}
void G1CaptureFrame(int width,int height){
    static int views=0;
    if(++views!=240)return;
    std::vector<unsigned char> pixels(size_t(width)*height*4);
    glReadPixels(0,0,width,height,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());
    std::ofstream out(assetsPath+"/frame.ppm",std::ios::binary);
    out<<"P6\n"<<width<<" "<<height<<"\n255\n";
    for(int y=height-1;y>=0;y--)for(int x=0;x<width;x++)out.write(reinterpret_cast<char*>(pixels.data()+4*(y*width+x)),3);
    __android_log_print(ANDROID_LOG_INFO,"G1Quest","Saved diagnostic eye frame %dx%d",width,height);
}
void G1PrepareFrame(){
    if(!sim)return;
    std::lock_guard<std::mutex> guard(mutex);
    mjvOption opt;mjv_defaultOption(&opt);opt.geomgroup[3]=0;
    mjvCamera cam;mjv_defaultCamera(&cam);
    mjv_updateScene(sim->model,sim->data,&opt,nullptr,&cam,mjCAT_ALL,&scene);
    static auto last=std::chrono::steady_clock::now();static int frames=0;
    frames++;
    double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-last).count();
    if(elapsed>=5){__android_log_print(ANDROID_LOG_INFO,"G1Quest","render %.1f fps, sim %.2fs",frames/elapsed,sim->data->time);frames=0;last=std::chrono::steady_clock::now();}
}
void G1Render(const float* vp){
    if(!sim)return;
    InitGraphics();glUseProgram(program);glDisable(GL_CULL_FACE);glEnable(GL_DEPTH_TEST);
    glUniformMatrix4fv(vpLocation,1,GL_FALSE,vp);
    // MuJoCo +Z up -> XR +Y up, anchored ahead of the initial headset pose.
    XrMatrix4x4f world{};std::copy(sceneTransform.begin(),sceneTransform.end(),world.m);
    int renderedTriangles=2;
    for(int i=0;i<scene.ngeom;i++){
        const mjvGeom& g=scene.geoms[i];if(g.rgba[3]<.01)continue;
        // The physical floor remains, but real surroundings are visible through it.
        if(g.type==mjGEOM_PLANE)continue;
        if(g.type!=mjGEOM_MESH && g.type!=mjGEOM_BOX && g.type!=mjGEOM_PLANE && g.type!=mjGEOM_SPHERE && g.type!=mjGEOM_ELLIPSOID && g.type!=mjGEOM_CYLINDER && g.type!=mjGEOM_CAPSULE)continue;
        auto& mesh=Geometry(g);
        renderedTriangles+=mesh.count/3;
        float size[3]={g.size[0],g.size[1],g.size[2]};
        if(g.type==mjGEOM_MESH)size[0]=size[1]=size[2]=1;
        if(g.type==mjGEOM_SPHERE)size[1]=size[2]=size[0];
        if(g.type==mjGEOM_CYLINDER || g.type==mjGEOM_CAPSULE){size[2]=size[1];size[1]=size[0];}
        if(g.type==mjGEOM_PLANE){size[0]=size[1]=15;size[2]=.005;}
        XrMatrix4x4f local{};local.m[15]=1;
        for(int row=0;row<3;row++)for(int col=0;col<3;col++)local.m[col*4+row]=g.mat[row*3+col]*size[col];
        for(int j=0;j<3;j++)local.m[12+j]=g.pos[j];
        if(g.type==mjGEOM_PLANE)local.m[14]-=.005f;
        XrMatrix4x4f model;XrMatrix4x4f_Multiply(&model,&world,&local);
        glUniformMatrix4fv(modelLocation,1,GL_FALSE,model.m);
        glUniform4fv(colorLocation,1,g.rgba);glBindVertexArray(mesh.vao);glDrawArrays(GL_TRIANGLES,0,mesh.count);
    }
    if(renderedTriangles>100000)throw std::runtime_error("Rendered scene exceeds 100000 triangle budget");
    static bool reported=false;
    if(!reported){__android_log_print(ANDROID_LOG_INFO,"G1Quest","Rendered scene: %d triangles per eye",renderedTriangles);reported=true;}
    glBindVertexArray(0);glUseProgram(0);
    // Card is authored in the original observer frame; move it with the scene.
    XrMatrix4x4f originalInverse{{0,1,0,0, 0,0,1,0, 1,0,0,0, 2.5,0,0,1}};
    XrMatrix4x4f cardWorld,cardVP,viewProjection;std::copy_n(vp,16,viewProjection.m);
    XrMatrix4x4f_Multiply(&cardWorld,&world,&originalInverse);
    XrMatrix4x4f_Multiply(&cardVP,&viewProjection,&cardWorld);
    DrawLegend(cardVP.m,assetsPath,trackingStatus.load()+6*recordingStatus.load());
}

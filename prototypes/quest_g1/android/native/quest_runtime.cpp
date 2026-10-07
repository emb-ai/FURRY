#include "view_frames.h"
#include "quest_runtime.h"
#include "simulation.h"
#include "meta_retarget.h"
#include "legend.h"
#include "recording.h"
#include "tracking_space.h"
#include "runtime_stats.h"
#include "stats_hud.h"
#include "camera_stream.h"
#include "camera_pose_servo.h"
#include "recording_export.h"
#include <deque>
#include <set>
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
struct InputSample{TrackingFrame frame;float left,right;int64_t received;};
std::mutex inputMutex,statsMutex;
std::deque<InputSample> pendingInputs;
std::vector<std::string> pendingEvents;
std::atomic<bool> spaceInvalidated{false};
std::unique_ptr<CameraStream> cameraStream;
CameraFusion cameraFusion;
std::atomic<uint64_t> droppedInputs{0};
RuntimeStats stats;
int pauseReason=0;
double renderFps=0,renderMs=0;uint64_t skippedSceneUpdates=0;
double ClockSeconds(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
std::ofstream telemetry,cameraRecording;
uint64_t cameraRecordedSequence=~0ull;
TrackingSpaceChanges spaceChanges;
EpisodeRecorder recorder;
std::unique_ptr<RecordingExport> exporter;
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
mjvScene scene{},workerScene{};
std::mutex geometryMutex;
std::vector<mjvGeom> publishedGeometry;
XrMatrix4x4f publishedCamera{},frameCamera{},headMatrix{};
XrMatrix4x4f publishedEgoRotation{},frameEgoRotation{},publishedEgoWorld{},frameEgoWorld{};
float publishedEgoScale=1,frameEgoScale=1;
bool publishedEgoCalibrated=false;
TrackingFrame frameTracking;
RuntimeStats frameStats;
double frameAgeMs=0,frameDrawMs=0,frameNow=0;
int frameStatus=0,frameExportStatus=0;
std::atomic<bool> firstPerson{false};
bool refreshEgoAnchor=true;
int egoCamera=-1,headBody=-1;
void PublishGeometry(){
    mjvOption option;mjv_defaultOption(&option);option.geomgroup[3]=0;
    mjvCamera camera;mjv_defaultCamera(&camera);
    mjv_updateScene(sim->model,sim->data,&option,nullptr,&camera,mjCAT_ALL,&workerScene);
    XrMatrix4x4f pose{};pose.m[15]=1;
    for(int r=0;r<3;r++){
        for(int c=0;c<3;c++)pose.m[4*c+r]=float(sim->data->cam_xmat[9*egoCamera+3*r+c]);
        pose.m[12+r]=float(sim->data->cam_xpos[3*egoCamera+r]);
    }
    std::lock_guard<std::mutex> guard(geometryMutex);
    publishedGeometry.assign(workerScene.geoms,workerScene.geoms+workerScene.ngeom);
    publishedCamera=pose;publishedEgoCalibrated=retarget->calibrated;
    if(publishedEgoCalibrated){
        const double* basis=retarget->StageToRobotRotation();
        publishedEgoRotation={};publishedEgoRotation.m[15]=1;
        for(int r=0;r<3;r++)for(int c=0;c<3;c++)publishedEgoRotation.m[4*c+r]=float(basis[3*c+r]);
        publishedEgoScale=float(retarget->VisualScale());
        publishedEgoWorld=CalibratedCameraWorld(PoseMatrix(retarget->CameraCalibrationPose()),
            PoseMatrix(retarget->HeadCalibrationPose()),publishedEgoRotation,publishedEgoScale);
    }
}
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
    std::ifstream cameraConfig(assets+"/camera_stream_url.txt");std::string cameraUrl;std::getline(cameraConfig,cameraUrl);
    cameraStream=std::make_unique<CameraStream>(cameraUrl);
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
    mjv_defaultScene(&workerScene);mjv_makeScene(sim->model,&workerScene,2048);
    egoCamera=mj_name2id(sim->model,mjOBJ_CAMERA,"ego");
    headBody=mj_name2id(sim->model,mjOBJ_BODY,"head_link");
    if(egoCamera<0)throw std::runtime_error("Missing ego camera");
    publishedGeometry.reserve(2048);PublishGeometry();
    telemetry.open(assets+"/runtime_stats.csv",std::ios::trunc);
    telemetry<<"wall_s,sim_s,cycle_ms,physics_ms,gmr_ms,inference_ms,rtf,contacts,overlap_points,pairs,max_depth_mm,solver_iterations,constraints,body_gap_ms,input_age_ms,body_confidence,body_version,state,reason,target_x,target_y,actual_x,actual_y,cmd_vx,cmd_vy,actual_vx,actual_vy,root_error,height,tilt,gmr_residual,gmr_iterations,overruns,dropped_inputs,left_foot_contacts,right_foot_contacts,leg_error_deg,physics_workers,camera_error_m,camera_error_deg,visual_scale,camera_connected,camera_state,camera_age_ms,camera_fit_mm,camera_legs,camera_weight,camera_clock_synced,camera_clock_offset_ms,camera_clock_rtt_ms\n";
    try{exporter=std::make_unique<RecordingExport>(app->activity->vm,app->activity->clazz);exporter->Latest(assets+"/recordings");}
    catch(const std::exception& e){__android_log_print(ANDROID_LOG_ERROR,"G1Quest","Recording export unavailable: %s",e.what());}
    running=true;
    worker=std::thread([]{
        double rateWall=ClockSeconds(),rateSim=0,lastTelemetry=0;RuntimeStats sample;sample.physicsWorkers=sim->physics_workers;
        auto next=std::chrono::steady_clock::now();
        while(running){
            double cycleStart=ClockSeconds();
            TrackingFrame input;std::chrono::steady_clock::time_point receivedAt;
            std::deque<InputSample> inputs;std::vector<std::string> events;
            {std::lock_guard<std::mutex> guard(inputMutex);input=latestTracking;receivedAt=trackingTime;inputs.swap(pendingInputs);events.swap(pendingEvents);}
            {
                std::lock_guard<std::mutex> guard(mutex);
                for(const auto& event:events){
                    if(event=="focus_pause"){wasApplying=false;retarget->Pause();sim->PauseWholeBodyReference();}
                    Record([&]{recorder.Event(event.c_str(),input.sequence);});
                }
                if(spaceInvalidated.exchange(false)){
                    trackingEnabled=false;retarget->calibrated=false;calibrate=false;toggleTracking=false;cameraFusion.Reset();
                    wasApplying=false;retarget->Pause();sim->PauseWholeBodyReference();pauseReason=1;
                    Record([&]{recorder.Event("reference_space_change",input.sequence);});
                }
                Record([&]{
                    for(const auto& entry:inputs)recorder.Input(entry.frame,entry.left,entry.right,entry.received);
                    if(toggleRecording.exchange(false)){
                        if(recorder.active){cameraRecording.close();recorder.Stop();if(exporter)exporter->Queue(recorder.path());__android_log_print(ANDROID_LOG_INFO,"G1Quest","Recording saved: %s",recorder.path().c_str());}
                        else{
                            recorder.Start(assetsPath,sim->model);
                            cameraRecording.open(std::filesystem::path(recorder.path())/"camera.jsonl");cameraRecordedSequence=~0ull;
                            recorder.Event(firstPerson?"ego_world":"observer_view",input.sequence);
                            auto received=std::chrono::duration_cast<std::chrono::nanoseconds>(receivedAt.time_since_epoch()).count();
                            recorder.Input(input,grip[0],grip[1],received);
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
                if(resetNow){cameraFusion.Reset();pauseReason=0;sim->Reset();wasApplying=false;trackingEnabled=false;retarget->calibrated=false;trackingStatus=0;calibrate=false;toggleTracking=false;faulted=false;lastReference={};grip[0]=0;grip[1]=0;Record([&]{recorder.Event("reset",input.sequence);});}
                bool valid=input.body.valid && (input.location_flags[0]&3)==3 && std::abs(input.xr_time_ns-input.body.time_ns)<200000000LL && std::chrono::steady_clock::now()-receivedAt<std::chrono::milliseconds(200);
                bool calibratedNow=false;
                if(calibrate.load() && valid){
                    cameraFusion.Reset();pauseReason=0;calibratedNow=true;Record([&]{recorder.Event("calibrate",input.sequence);});
                    retarget->Calibrate(sim->model,sim->data,input);trackingEnabled=true;calibrate=false;
                    __android_log_print(ANDROID_LOG_INFO,"G1Quest","Tracking calibrated: Meta full body -> native GMR -> TWIST2");
                }
                if(toggleTracking.exchange(false) && retarget->calibrated){trackingEnabled=!trackingEnabled;Record([&]{recorder.Event(trackingEnabled?"tracking_resume":"tracking_pause",input.sequence);});}
                if(retarget->calibrated && valid && !retarget->Compatible(input)){
                    trackingEnabled=false;retarget->calibrated=false;pauseReason=2;
                    Record([&]{recorder.Event("body_skeleton_changed",input.sequence);});
                }
                bool applyReference=trackingEnabled && valid && retarget->calibrated;
                retarget->EnableCameraTracking(firstPerson.load());
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
                    double gmrStart=ClockSeconds();
                    cameraFusion.Observe(input,CameraEpochMs());CameraSkeleton optical;
                    std::string opticalRaw;bool haveOptical=cameraStream->Latest(optical,&opticalRaw);
                    if(haveOptical && recorder.active && optical.sequence!=cameraRecordedSequence){
                        cameraRecording<<"{\"input_sequence\":"<<input.sequence<<",\"received_epoch_ms\":"<<std::setprecision(17)<<optical.receivedMs<<",\"clock_offset_ms\":"<<cameraStream->clockOffsetMs.load()<<",\"payload\":"<<opticalRaw<<"}\n";
                        cameraRecordedSequence=optical.sequence;
                    }
                    auto fused=cameraFusion.Apply(input,haveOptical?&optical:nullptr,CameraEpochMs());
                    auto whole=retarget->Solve(fused);
                    if(retarget->solver().HasCameraTarget()){
                        TrackedPose actual;std::copy_n(sim->data->cam_xpos+3*egoCamera,3,actual.position.begin());
                        mju_mat2Quat(actual.quaternion.data(),sim->data->cam_xmat+9*egoCamera);
                        whole=CameraPoseServo(whole,retarget->solver().CameraTarget(),actual,sim->data->qpos+3);
                    }
                    sample.gmrMs=(ClockSeconds()-gmrStart)*1000;
                    double u=std::clamp((sim->data->time-blendStarted)/.5,0.,1.);
                    for(int j=0;j<35;j++)whole[j]=blendOrigin[j]+u*(whole[j]-blendOrigin[j]);
                    sim->SetWholeBodyReference(whole);std::copy_n(whole.begin()+6,29,lastReference.begin());
                }else{sample.gmrMs=0;sim->PauseWholeBodyReference();retarget->Pause();}
                wasApplying=applyReference;
                trackingStatus=!valid?3:retarget->calibrated?(trackingEnabled?(valid?(retarget->error>.3?5:1):3):2):0;
                Record([&]{recorder.Frame(sim->model,sim->data,sim->steps,input,resetNow,calibratedNow,applyReference,valid,
                    leftGrip,rightGrip,trackingStatus,retarget->error,(retarget->error>.3),lastReference);});
                Record([&]{recorder.Mimic(input,sim->steps,sim->data->time,sim->WholeBodyReference());});
                // Keep the actual input episode for deterministic policy replay.
                if(trace && traceRows++<60000){
                    trace<<sim->steps<<","<<sim->data->time<<","<<resetNow<<","<<applyReference<<","<<leftGrip<<","<<rightGrip;
                    for(auto q:lastReference)trace<<","<<q;
                    trace<<","<<valid<<","<<trackingStatus.load()<<","<<retarget->error<<","<<(retarget->error>.3)<<","<<sim->data->qpos[2];
                    for(int i=3;i<7;i++)trace<<","<<sim->data->qpos[i];
                    for(const auto& pose:{input.head,input.hands[0],input.hands[1]}){
                        for(auto p:pose.position)trace<<","<<p;for(auto q:pose.quaternion)trace<<","<<q;
                    }
                    trace<<"\n";if(sim->steps%1000==0)trace.flush();
                }
                double physicsStart=ClockSeconds();
                for(int i=0;i<10;i++)sim->Step(false,leftGrip,rightGrip);
                PublishGeometry();
                sample.physicsMs=std::max(0.,(ClockSeconds()-physicsStart)*1000-sim->inference_ms);
                sample.inferenceMs=sim->inference_ms;sample.cycleMs=(ClockSeconds()-cycleStart)*1000;
                if(sample.cycleMs>10)sample.overruns++;
                sample.published=ClockSeconds();sample.simTime=sim->data->time;
                if(sample.simTime<rateSim){rateSim=sample.simTime;rateWall=sample.published;}
                if(sample.published-rateWall>=.5){sample.realTimeFactor=(sample.simTime-rateSim)/(sample.published-rateWall);rateSim=sample.simTime;rateWall=sample.published;}
                sample.contacts=sim->data->ncon;sample.constraints=sim->data->nefc;sample.overlaps=0;sample.depthMm=0;
                std::set<std::pair<int,int>> contactPairs;
                for(int i=0;i<sim->data->ncon;i++){const auto& c=sim->data->contact[i];contactPairs.emplace(std::min(c.geom1,c.geom2),std::max(c.geom1,c.geom2));if(c.dist<0){sample.overlaps++;sample.depthMm=std::max(sample.depthMm,-1000*c.dist);}}
                sample.footContacts={};
                for(int i=0;i<sim->data->ncon;i++){
                    const auto& c=sim->data->contact[i];
                    for(int side=0;side<2;side++){
                        int foot=mj_name2id(sim->model,mjOBJ_BODY,side?"right_ankle_roll_link":"left_ankle_roll_link");
                        int a=sim->model->geom_bodyid[c.geom1],b=sim->model->geom_bodyid[c.geom2];
                        if(c.efc_address>=0 && ((a==foot && b==0)||(b==foot && a==0)))sample.footContacts[side]++;
                    }
                }
                double legError=0;auto* gm=retarget->solver().model();
                for(int j=1;j<=12;j++){
                    int actual=mj_name2id(sim->model,mjOBJ_JOINT,mj_id2name(gm,mjOBJ_JOINT,j));
                    double e=sim->data->qpos[sim->model->jnt_qposadr[actual]]-sim->WholeBodyReference()[6+gm->jnt_qposadr[j]-7];
                    legError+=e*e;
                }
                sample.legErrorDegrees=std::sqrt(legError/12)*180/3.141592653589793;
                sample.pairs=contactPairs.size();sample.solverIterations=0;
                for(int i=0;i<std::min(mjNISLAND,std::max(1,sim->data->nisland));i++)sample.solverIterations+=sim->data->solver_niter[i];
                sample.warnings=0;for(const auto& w:sim->data->warning)sample.warnings+=w.number;
                sample.height=sim->data->qpos[2];double qx=sim->data->qpos[4],qy=sim->data->qpos[5];sample.tilt=std::acos(std::clamp(1-2*(qx*qx+qy*qy),-1.,1.))*180/3.141592653589793;
                sample.inputAgeMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-receivedAt).count();
                sample.bodyGapMs=(input.xr_time_ns-input.body.time_ns)*1e-6;sample.confidence=input.body.confidence;sample.bodyVersion=input.body.skeleton_version;
                sample.residual=retarget->error;sample.gmrIterations=retarget->solver().iterations;
                for(int a=0;a<2;a++){sample.targetXY[a]=retarget->solver().data()->qpos[a];sample.actualXY[a]=sim->data->qpos[a];sample.commandXY[a]=sim->WholeBodyReference()[a];sample.velocityXY[a]=sim->data->qvel[a];}
                CameraSkeleton displayCamera;bool haveDisplayCamera=cameraStream->Latest(displayCamera);
                sample.cameraOverlay=cameraFusion.MapForDisplay(haveDisplayCamera?&displayCamera:nullptr,input,CameraEpochMs());
                sample.cameraState=cameraFusion.stats.state;sample.cameraLegs=cameraFusion.stats.legs;
                sample.cameraAgeMs=cameraFusion.stats.ageMs;sample.cameraFitMm=cameraFusion.stats.fitMm;
                sample.cameraWeight=cameraFusion.stats.weight;sample.cameraConnected=cameraStream->connected;
                sample.cameraClockSynced=cameraStream->clockSynced;sample.cameraClockOffsetMs=cameraStream->clockOffsetMs;sample.cameraClockRttMs=cameraStream->clockRttMs;
                sample.visualScale=retarget->calibrated?retarget->VisualScale():1;
                sample.cameraPositionError=sample.cameraOrientationError=-1;
                if(retarget->solver().HasCameraTarget()){
                    const auto& goal=retarget->solver().CameraTarget();double delta[3],actual[4],inverse[4],relative[4],angle[3];
                    mju_sub3(delta,goal.position.data(),sim->data->cam_xpos+3*egoCamera);
                    sample.cameraPositionError=mju_norm3(delta);
                    mju_mat2Quat(actual,sim->data->cam_xmat+9*egoCamera);mju_negQuat(inverse,actual);
                    mju_mulQuat(relative,inverse,goal.quaternion.data());mju_quat2Vel(angle,relative,1);
                    sample.cameraOrientationError=mju_norm3(angle)*180/3.141592653589793;
                }
                sample.positionError=std::hypot(sample.targetXY[0]-sample.actualXY[0],sample.targetXY[1]-sample.actualXY[1]);sample.status=trackingStatus;sample.reason=pauseReason;sample.droppedInputs=droppedInputs;
                {std::lock_guard<std::mutex> guard(statsMutex);stats=sample;}
                if(sample.published-lastTelemetry>.1){
                    const auto& x=sample;
                    telemetry<<x.published<<','<<x.simTime<<','<<x.cycleMs<<','<<x.physicsMs<<','<<x.gmrMs<<','<<x.inferenceMs<<','<<x.realTimeFactor<<','<<x.contacts<<','<<x.overlaps<<','<<x.pairs<<','<<x.depthMm<<','<<x.solverIterations<<','<<x.constraints<<','<<x.bodyGapMs<<','<<x.inputAgeMs<<','<<x.confidence<<','<<x.bodyVersion<<','<<x.status<<','<<x.reason;
                    for(const auto& values:{x.targetXY,x.actualXY,x.commandXY,x.velocityXY})for(double v:values)telemetry<<','<<v;
                    telemetry<<','<<x.positionError<<','<<x.height<<','<<x.tilt<<','<<x.residual<<','<<x.gmrIterations<<','<<x.overruns<<','<<x.droppedInputs<<','<<x.footContacts[0]<<','<<x.footContacts[1]<<','<<x.legErrorDegrees<<','<<x.physicsWorkers<<','<<x.cameraPositionError<<','<<x.cameraOrientationError<<','<<x.visualScale<<','<<x.cameraConnected<<','<<x.cameraState<<','<<x.cameraAgeMs<<','<<x.cameraFitMm<<','<<x.cameraLegs<<','<<x.cameraWeight<<','<<x.cameraClockSynced<<','<<x.cameraClockOffsetMs<<','<<x.cameraClockRttMs<<'\n';lastTelemetry=x.published;
                    if(sim->steps%1000==0)telemetry.flush();
                }
                if(sim->steps%5000==0)__android_log_print(ANDROID_LOG_INFO,"G1Quest","sim=%.2fs height=%.3f inference=%.3fms",sim->data->time,sim->data->qpos[2],sim->inference_ms);
                if(sim->steps%5000==0)__android_log_print(ANDROID_LOG_INFO,"G1Quest","tracking state=%d GMR task residual=%.3f",trackingStatus.load(),retarget->error);
            }catch(const std::exception& e){
                __android_log_print(ANDROID_LOG_ERROR,"G1Quest","Simulation paused at %.3fs: %s; trace saved",sim->data->time,e.what());
                std::lock_guard<std::mutex> guard(mutex);trace.flush();Record([&]{recorder.Event("simulation_fault",input.sequence);recorder.Flush();});faulted=true;trackingEnabled=false;calibrate=false;trackingStatus=4;pauseReason=3;
                {std::lock_guard<std::mutex> statsGuard(statsMutex);stats.status=4;stats.reason=3;}
                telemetry.flush();
            }
            next+=std::chrono::milliseconds(10);std::this_thread::sleep_until(next);
            if(std::chrono::steady_clock::now()-next>std::chrono::milliseconds(100))next=std::chrono::steady_clock::now();
        }
    });
    __android_log_print(ANDROID_LOG_INFO,"G1Quest","Initialized autonomous MuJoCo %s, %d actuators, %d physics workers",mj_versionString(),sim->model->nu,sim->physics_workers);
}
void G1Shutdown(bool interrupted){
    running=false;if(worker.joinable())worker.join();
    Record([&]{
        if(interrupted){recorder.Event("app_error",latestTracking.sequence);recorder.Flush();recorder.Abort();}
        else {bool wasRecording=recorder.active;recorder.Stop("app_shutdown");if(wasRecording && exporter)exporter->Queue(recorder.path());}
    });
    exporter.reset();cameraStream.reset();
    cameraRecording.close();telemetry.close();trace.close();mjv_freeScene(&scene);mjv_freeScene(&workerScene);retarget.reset();sim.reset();
}
void G1SetActive(bool value){
    if(active.exchange(value)!=value){std::lock_guard<std::mutex> guard(inputMutex);pendingEvents.emplace_back(value?"focus_resume":"focus_pause");}
}
void G1Grip(int hand,float value){if(hand>=0 && hand<2)grip[hand]=value;}
void G1Reset(){reset=true;placeScene=true;refreshEgoAnchor=true;}
void G1ReferenceSpaceChange(int64_t changeTime){
    std::lock_guard<std::mutex> guard(inputMutex);
    spaceChanges.Schedule(changeTime);
}
void G1SubmitTracking(const TrackingFrame& input){
    if(cameraStream)cameraStream->Submit(input);
    {std::lock_guard<std::mutex> guard(inputMutex);latestTracking=input;trackingTime=std::chrono::steady_clock::now();
        if(spaceChanges.Advance(input.xr_time_ns)){spaceInvalidated=true;placeScene=true;refreshEgoAnchor=true;}
        if(pendingInputs.size()>=512){pendingInputs.pop_front();droppedInputs++;}
        pendingInputs.push_back({input,grip[0].load(),grip[1].load(),EpisodeRecorder::Now()});
    }
    if((input.location_flags[0]&3)==3 && placeScene.exchange(false)){
        double rotation[9];mju_quat2Mat(rotation,input.head.quaternion.data());
        double fx=-rotation[2],fz=-rotation[8],norm=std::hypot(fx,fz);
        if(norm<.2){fx=0;fz=-1;norm=1;}fx/=norm;fz/=norm;
        sceneTransform={float(-fx),0,float(-fz),0, float(-fz),0,float(fx),0, 0,1,0,0,
                        float(input.head.position[0]+3.0*fx),0,float(input.head.position[2]+3.0*fz),1};
        __android_log_print(ANDROID_LOG_INFO,"G1Quest","Scene placed 3.0m ahead of headset on stage floor");
    }
}
void G1Calibrate(){calibrate=true;}
void G1ToggleView(){firstPerson=!firstPerson.load();refreshEgoAnchor=true;std::lock_guard<std::mutex>g(inputMutex);pendingEvents.emplace_back(firstPerson?"ego_world":"observer_view");}
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
    static double last=ClockSeconds();static int frames=0;
    frames++;double now=ClockSeconds();frameNow=now;if(now-last>=.5){renderFps=frames/(now-last);frames=0;last=now;}
    {std::lock_guard<std::mutex> guard(inputMutex);frameTracking=latestTracking;
        frameAgeMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-trackingTime).count();}
    {std::unique_lock<std::mutex> guard(statsMutex,std::try_to_lock);if(guard.owns_lock())frameStats=stats;}
    frameDrawMs=renderMs;frameStatus=trackingStatus.load();frameExportStatus=exporter?exporter->status.load():3;
    headMatrix=PoseMatrix(frameTracking.head);

    std::unique_lock<std::mutex> guard(geometryMutex,std::try_to_lock);
    if(!guard.owns_lock()){skippedSceneUpdates++;return;}
    scene.ngeom=std::min(int(publishedGeometry.size()),scene.maxgeom);
    std::copy_n(publishedGeometry.begin(),scene.ngeom,scene.geoms);frameCamera=publishedCamera;
    if(publishedEgoCalibrated){frameEgoRotation=publishedEgoRotation;frameEgoScale=publishedEgoScale;frameEgoWorld=publishedEgoWorld;refreshEgoAnchor=false;}
    else if(refreshEgoAnchor && (frameTracking.location_flags[0]&3)==3){
        frameEgoRotation=LevelEgoRotation(frameCamera,headMatrix);
        frameEgoScale=std::clamp(headMatrix.m[13]/std::max(.5f,frameCamera.m[14]),.7f,2.f);
        frameEgoWorld=CalibratedCameraWorld(frameCamera,headMatrix,frameEgoRotation,frameEgoScale);refreshEgoAnchor=false;
    }
}
void G1Render(const float* vp,const float* projection){
    double renderStart=ClockSeconds();
    if(!sim)return;
    InitGraphics();glUseProgram(program);glDisable(GL_CULL_FACE);glEnable(GL_DEPTH_TEST);
    glUniformMatrix4fv(vpLocation,1,GL_FALSE,vp);
    // MuJoCo +Z up -> XR +Y up, anchored ahead of the initial headset pose.
    XrMatrix4x4f world{};std::copy(sceneTransform.begin(),sceneTransform.end(),world.m);
    if(firstPerson)world=frameEgoWorld;
    int renderedTriangles=2;
    for(int i=0;i<scene.ngeom;i++){
        const mjvGeom& g=scene.geoms[i];if(g.rgba[3]<.01)continue;
        if(firstPerson && g.objtype==mjOBJ_GEOM && g.objid>=0 &&
           sim->model->geom_bodyid[g.objid]==headBody)continue;
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
    if(!firstPerson)DrawLegend(cardVP.m,assetsPath,frameStatus+6*recordingStatus.load());
    XrMatrix4x4f viewProjectionHud,hudVP;std::copy_n(vp,16,viewProjectionHud.m);
    XrMatrix4x4f_Multiply(&hudVP,&viewProjectionHud,&headMatrix);
    renderedTriangles+=DrawStatsHud(hudVP.m,assetsPath,frameStats,renderFps,frameDrawMs,skippedSceneUpdates,frameNow,frameStatus,active.load(),frameExportStatus,frameTracking,frameAgeMs,firstPerson.load(),firstPerson?frameEgoScale:1.f);
    if(renderedTriangles>100000)throw std::runtime_error("Scene and HUD exceed triangle budget");
    renderMs=.9*renderMs+.1*(ClockSeconds()-renderStart)*1000;
}

#include "view_frames.h"
#include "quest_runtime.h"
#include "simulation.h"
#include "meta_retarget.h"
#ifdef G1_ENABLE_FOOT_FLOOR_GUARD
#include "foot_floor.h"
#endif
#include "recording.h"
#include "tracking_space.h"
#include "runtime_stats.h"
#include "stats_hud.h"
#include "operator_skeleton_draw.h"
#include "capture_hud.h"
#include "camera_stream.h"
#include "world_skeleton_gl.h"
#include "camera_pose_servo.h"
#include "recording_export.h"
#include "quest_menu_gl.h"
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
#ifdef G1_ENABLE_FOOT_FLOOR_GUARD
std::unique_ptr<FootFloorGuard> floorGuard;
#endif
std::array<float,35> blendOrigin{};
double blendStarted=0;
bool wasApplying=false;
TrackingFrame latestTracking;
struct InputSample{TrackingFrame frame;float left,right;int64_t received;};
std::mutex inputMutex,statsMutex;
std::deque<InputSample> pendingInputs;
std::vector<std::string> pendingEvents;
struct MenuCommand { questmenu::Action action; int policyIndex=-1; };
std::deque<MenuCommand> pendingMenuActions;
questmenu::State menuState;
std::vector<questpolicy::Entry> policyCatalog;
std::atomic<int> selectedPolicy{0};
std::atomic<bool> catchUpEnabled{true};
std::atomic<int> sessionMode{0},sessionCapture{0},sessionPlan{0},sessionScene{1},requestedScene{-1};
std::atomic<bool> menuOpen{true},userPaused{true},calibrationReady{false},anchorMenu{true};
std::atomic<bool> passthroughVisible{true},debugEnabled{false},debugStats{true},debugMeta{false},debugCamera{false},debugTargets{false},debugContacts{false};
std::atomic<bool> refreshGeometry{false},settingsDirty{false};
std::atomic<double> acceptedRecordingSeconds{0};
std::atomic<double> recordedSeconds{0};
XrMatrix4x4f menuWorld{};
struct MenuRay {TrackedPose pose;bool valid=false,active=false;float trigger=0;};
std::array<MenuRay,2> menuRays;
std::array<questmenu::TriggerLatch,2> menuTrigger;
std::vector<questmenu::Pointer> framePointers;
std::string menuNotice;
bool SessionPaused(){return menuOpen.load()||userPaused.load();}
bool HumanCapture(){return sessionMode.load()==1&&sessionCapture.load()==1;}
void Notice(const std::string& value){std::lock_guard<std::mutex> guard(inputMutex);menuNotice=value;}
void QueueMenuAction(questmenu::Action action,int policyIndex=-1){std::lock_guard<std::mutex> guard(inputMutex);pendingMenuActions.push_back({action,policyIndex});}
std::atomic<bool> spaceInvalidated{false};
std::unique_ptr<CameraStream> cameraStream;
CameraFusion cameraFusion;
std::atomic<uint64_t> droppedInputs{0};
RuntimeStats stats;
GuidedCapture capture,publishedCapture,frameCapture;
bool publishedCaptureFresh=false,frameCaptureFresh=false;
std::ofstream captureTimeline;
int pauseReason=0;
double renderFps=0,renderMs=0;uint64_t skippedSceneUpdates=0;
double ClockSeconds(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
std::ofstream telemetry,cameraRecording;
uint64_t cameraRecordedSequence=~0ull;
TrackingSpaceChanges spaceChanges;
EpisodeRecorder recorder;
std::unique_ptr<mjData,decltype(&mj_deleteData)> calibrationSnapshot{nullptr,mj_deleteData};
TrackingFrame calibrationInput;
int calibrationStep=0;
int64_t calibrationReceived=0;
std::array<float,2> calibrationGrip{};
std::array<float,29> calibrationReference{};
std::array<float,35> calibrationMimic{};
std::unique_ptr<RecordingExport> exporter;
std::atomic<int> recordingStatus{0};
std::atomic<double> recordingStarted{0};
std::atomic<bool> toggleRecording{false};
bool recordingShortcut=false,restartAfterCalibration=false;
std::chrono::steady_clock::time_point trackingTime;
std::atomic<bool> calibrate{false};
bool trackingEnabled=false;
std::atomic<bool> faulted{false};
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
CameraOverlay frameOperatorCamera;
WorldSkeleton publishedTargets,frameTargets,frameMenuRays;
double frameAgeMs=0,frameDrawMs=0,frameNow=0;
int frameStatus=0,frameExportStatus=0;
std::atomic<bool> firstPerson{false};
std::atomic<bool> refreshEgoAnchor{true};
int egoCamera=-1,headBody=-1;
void PublishGeometry(){
    mjvOption option;mjv_defaultOption(&option);option.geomgroup[3]=debugEnabled&&debugContacts;
    // Collision inspection replaces the exterior shell so contacts remain
    // visible and this diagnostic mode has ample room in the render budget.
    option.geomgroup[1]=!(debugEnabled&&debugContacts);
    option.flags[mjVIS_CONTACTPOINT]=debugEnabled&&debugContacts;
    option.flags[mjVIS_CONTACTFORCE]=debugEnabled&&debugContacts;
    mjvCamera camera;mjv_defaultCamera(&camera);
    mjv_updateScene(sim->model,sim->data,&option,nullptr,&camera,mjCAT_ALL,&workerScene);
    XrMatrix4x4f pose{};pose.m[15]=1;
    for(int r=0;r<3;r++){
        for(int c=0;c<3;c++)pose.m[4*c+r]=float(sim->data->cam_xmat[9*egoCamera+3*r+c]);
        pose.m[12+r]=float(sim->data->cam_xpos[3*egoCamera+r]);
    }
    std::lock_guard<std::mutex> guard(geometryMutex);
    publishedGeometry.assign(workerScene.geoms,workerScene.geoms+workerScene.ngeom);
    publishedTargets={};
    if(retarget->calibrated&&debugEnabled&&debugTargets){
        auto* gm=retarget->solver().model();auto* gd=retarget->solver().data();
        const std::array<float,4> tint{.7f,.9f,.35f,.8f};
        for(int b=1;b<gm->nbody;b++){
            std::array<double,3> point;std::copy_n(gd->xpos+3*b,3,point.begin());
            publishedTargets.marks.push_back({point,.008f,tint});
            int parent=gm->body_parentid[b];if(parent>0){std::array<double,3> start;std::copy_n(gd->xpos+3*parent,3,start.begin());publishedTargets.bones.push_back({start,point,.002f,tint});}
        }
    }
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

void CloseFailedRecordingStreams()noexcept{
    for(auto* stream:{&cameraRecording,&captureTimeline}){
        stream->exceptions(std::ios::goodbit);
        if(stream->is_open())stream->close();
        stream->clear();
    }
}
void FinishRecording(uint64_t sequence,const char* reason="user_stop"){
    if(!recorder.active)return;
    if(capture.running){
        acceptedRecordingSeconds=capture.Total();
        recorder.Event(capture.completed?"guide_complete":"guide_partial_stop",sequence);
        captureTimeline.flush();captureTimeline.close();capture.WriteSummary(recorder.path());capture.Stop();
    }
    recordedSeconds=std::max(0.,ClockSeconds()-recordingStarted.load());
    std::ofstream summary;summary.exceptions(std::ios::badbit|std::ios::failbit);summary.open(recorder.path()+"/episode_summary.json");
    summary<<std::setprecision(17)<<"{\"schema_version\":1,\"recorded_seconds\":"<<recordedSeconds.load()
           <<",\"accepted_seconds\":"<<acceptedRecordingSeconds.load()<<",\"pause_intervals_excluded\":true}\n";summary.close();
    if(cameraRecording.is_open())cameraRecording.close();
    recorder.Stop(reason);if(exporter)exporter->Queue(recorder.path());
    userPaused=true;menuOpen=true;anchorMenu=true;Notice("Запись сохранена.");
}
template<class F> void Record(F work){
    try{work();recordingStatus=recorder.active?1:(recorder.failed?2:0);}
    catch(const std::exception& e){recorder.Abort();CloseFailedRecordingStreams();recordingStatus=2;__android_log_print(ANDROID_LOG_ERROR,"G1Quest","Recording failed: %s",e.what());}
}
void StoreCalibration(const TrackingFrame& input,int64_t received){
    calibrationSnapshot.reset(mj_copyData(nullptr,sim->model,sim->data));
    if(!calibrationSnapshot)throw std::runtime_error("Could not save calibration snapshot");
    calibrationInput=input;calibrationReceived=received;calibrationStep=sim->steps;
    calibrationGrip={grip[0].load(),grip[1].load()};calibrationReference=lastReference;calibrationMimic=sim->WholeBodyReference();
}
void RecordCalibration(){
    if(!calibrationSnapshot)throw std::runtime_error("Missing calibration snapshot");
    recorder.Event("calibrate",calibrationInput.sequence);
    recorder.Frame(sim->model,calibrationSnapshot.get(),calibrationStep,calibrationInput,false,true,false,true,
                   calibrationGrip[0],calibrationGrip[1],3,retarget->error,false,calibrationReference);
    recorder.Mimic(calibrationInput,calibrationStep,calibrationSnapshot->time,calibrationMimic);
}

void CopyAssets(AAssetManager* manager, const std::string& prefix, const std::filesystem::path& dest) {
    std::filesystem::create_directories(dest);
    AAssetDir* dir=AAssetManager_openDir(manager,prefix.c_str());
    if(!dir) throw std::runtime_error("Cannot open assets directory");
    while(const char* name=AAssetDir_getNextFileName(dir)) {
        std::string path=prefix.empty()?name:prefix+"/"+name;
        // Preserve a device's configured camera endpoint across APK updates.
        if(path=="camera_stream_url.txt" && std::filesystem::exists(dest/name) && std::filesystem::file_size(dest/name)>0)continue;
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
Mesh& Geometry(const mjvGeom& g,bool collision=false){
    // mjvGeom encodes mesh i as 2*i (or 2*i+1 for its convex hull).
    int id=g.dataid/2;
    int key=g.type==mjGEOM_MESH ? id+(collision?100000:0) : -1-g.type;
    auto it=meshes.find(key); if(it!=meshes.end())return it->second;
    std::vector<float> v;
    if(g.type==mjGEOM_MESH){
        if(!collision && id>=0 && id<static_cast<int>(visualMeshes.size()))
            return meshes.emplace(key,Upload(visualMeshes[id])).first->second;
        const mjModel* m=sim->model;
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
std::vector<std::vector<float>> LoadVisualMeshes(const std::string& name,const mjModel* model){
    std::ifstream visual(assetsPath+"/visual_meshes-"+name+".bin",std::ios::binary);
    uint32_t count=0;visual.read(reinterpret_cast<char*>(&count),4);
    if(!visual || count!=uint32_t(model->nmesh))throw std::runtime_error("Visual mesh count mismatch");
    std::vector<std::vector<float>> out(count);
    for(auto& mesh:out){uint32_t n=0;visual.read(reinterpret_cast<char*>(&n),4);if(n>3000000)throw std::runtime_error("Invalid visual mesh");mesh.resize(size_t(n)*6);visual.read(reinterpret_cast<char*>(mesh.data()),mesh.size()*sizeof(float));}
    if(!visual)throw std::runtime_error("Truncated visual mesh asset");
    return out;
}
void LoadPreferences(){
    selectedPolicy=0;catchUpEnabled=true;
    std::ifstream input(assetsPath+"/menu_settings.txt");
    int version,mode,sc,kind,plan,view,pass,enabled,ds,dm,dc,dt,dcontacts;
    if(input>>version>>mode>>sc>>kind>>plan>>view>>pass>>enabled>>ds>>dm>>dc>>dt>>dcontacts && version==1 && mode>=0&&mode<2 && sc>=0&&sc<3 && kind>=0&&kind<2 && plan>=0&&plan<2){
        sessionMode=mode;sessionScene=sc;sessionCapture=kind;sessionPlan=plan;firstPerson=view!=0;passthroughVisible=pass!=0;
        debugEnabled=enabled!=0;debugStats=ds!=0;debugMeta=dm!=0;debugCamera=dc!=0;debugTargets=dt!=0;debugContacts=dcontacts!=0;
    }
    std::string savedPolicy;
    if(input>>savedPolicy)for(size_t i=0;i<policyCatalog.size();i++)if(policyCatalog[i].id==savedPolicy)selectedPolicy=int(i);
    int savedCatchUp;if(input>>savedCatchUp && (savedCatchUp==0||savedCatchUp==1))catchUpEnabled=savedCatchUp!=0;
    capture.mode=HumanCapture()?sessionPlan.load()+1:0;
    if(HumanCapture())firstPerson=false;
}
void SavePreferences(){
    std::ofstream output(assetsPath+"/menu_settings.txt",std::ios::trunc);
    output<<"1 "<<sessionMode<<' '<<sessionScene<<' '<<sessionCapture<<' '<<sessionPlan<<' '<<firstPerson<<' '<<passthroughVisible<<' '<<debugEnabled<<' '<<debugStats<<' '<<debugMeta<<' '<<debugCamera<<' '<<debugTargets<<' '<<debugContacts<<' '<<policyCatalog.at(selectedPolicy.load()).id<<' '<<catchUpEnabled<<'\n';
}
void ChangeConfiguration(questmenu::Action action){
    using A=questmenu::Action;
    if(recorder.active){Notice("Сначала сохраните запись.");return;}
    restartAfterCalibration=false;
    if(action==A::ModeSimulation)sessionMode=0;
    if(action==A::ModeTrajectories)sessionMode=1;
    if(action==A::CaptureRobot)sessionCapture=0;
    if(action==A::CaptureHuman)sessionCapture=1;
    if(action==A::PlanTrain)sessionPlan=0;
    if(action==A::PlanTest)sessionPlan=1;
    capture.mode=HumanCapture()?sessionPlan.load()+1:0;capture.Invalidate();
    if(HumanCapture()){firstPerson=false;refreshEgoAnchor=true;sim->Reset();faulted=false;lastReference={};}
    retarget->calibrated=false;calibrationReady=false;trackingEnabled=false;wasApplying=false;
    retarget->Pause();sim->PauseWholeBodyReference();userPaused=true;menuOpen=true;
    settingsDirty=true;refreshGeometry=true;Notice("");
}
void ApplyMenuAction(questmenu::Action action,bool fresh,uint64_t sequence,int policyIndex=-1){
    using A=questmenu::Action;
    if(action==A::CatchUp){
        if(recorder.active){Notice("Сначала сохраните запись.");return;}
        if(HumanCapture())return;
        catchUpEnabled=!catchUpEnabled.load();settingsDirty=true;restartAfterCalibration=false;
        reset=true;userPaused=true;menuOpen=true;return;
    }
    int policyRow=questmenu::PolicyRow(action);
    if(policyRow>=0){
        if(recorder.active){Notice("Сначала сохраните запись.");return;}
        if(HumanCapture())return;
        // Selection carries the catalog index from the UI, including page.
        int index=policyIndex;
        if(index<0||index>=int(policyCatalog.size())||index==selectedPolicy.load())return;
        userPaused=true;menuOpen=true;restartAfterCalibration=false;
        try{
            sim->SelectPolicy(questpolicy::VerifiedPath(assetsPath,policyCatalog[index]));
            selectedPolicy=index;settingsDirty=true;
            calibrationSnapshot.reset();retarget->calibrated=false;retarget->Pause();calibrationReady=false;
            trackingEnabled=false;wasApplying=false;faulted=false;lastReference={};grip[0]=0;grip[1]=0;
            calibrate=false;toggleRecording=false;cameraFusion.Reset();capture.Invalidate();trackingStatus=0;pauseReason=0;
            refreshGeometry=true;refreshEgoAnchor=true;
            {std::lock_guard<std::mutex> statsGuard(statsMutex);stats={};stats.physicsWorkers=sim->physics_workers;}
            Notice("Политика выбрана. Калибруйте кнопкой A.");
            __android_log_print(ANDROID_LOG_INFO,"G1Quest","Selected policy %s sha256=%s",policyCatalog[index].id.c_str(),policyCatalog[index].sha256.c_str());
        }catch(const std::exception& e){Notice("Ошибка загрузки. Сохранена прежняя политика.");__android_log_print(ANDROID_LOG_ERROR,"G1Quest","Policy selection: %s",e.what());}
        return;
    }
    if(action==A::ModeSimulation||action==A::ModeTrajectories||action==A::CaptureRobot||action==A::CaptureHuman||action==A::PlanTrain||action==A::PlanTest){ChangeConfiguration(action);return;}
    if(action==A::SceneEmpty||action==A::SceneCup||action==A::ScenePushT){
        if(recorder.active){Notice("Сначала сохраните запись.");return;}
        int selected=action==A::SceneEmpty?0:action==A::SceneCup?1:2;
        if(selected!=sessionScene.load()){restartAfterCalibration=false;requestedScene=selected;userPaused=true;menuOpen=true;Notice("Загрузка сцены...");}
        return;
    }
    if(action==A::Calibrate){calibrate=true;return;}
    if(action==A::ToggleRecording){
        if(recorder.active){recordingShortcut=true;toggleRecording=true;return;}
        if(!fresh||!retarget->calibrated||faulted||requestedScene>=0){Notice("Для записи нужны свежий трекинг и калибровка.");return;}
        // Y can start recording the current simulation without invalidating
        // the calibration or resetting the physical robot/policy history.
        if(sessionMode==0){sessionMode=1;sessionCapture=0;capture.mode=0;settingsDirty=true;}
        recordingShortcut=true;toggleRecording=true;return;
    }
    if(action==A::QuickReset){
        if(requestedScene>=0)return;
        if(recorder.active){Record([&]{FinishRecording(sequence,"user_stop");});if(recorder.failed)return;}
        toggleRecording=false;recordingShortcut=false;
        reset=true;restartAfterCalibration=true;userPaused=true;menuOpen=false;return;
    }
    if(action==A::Start){
        if(!fresh||!retarget->calibrated||faulted||requestedScene>=0){Notice("Нужны свежий трекинг и калибровка.");return;}
        if(sessionMode==1&&!recorder.active){recordingShortcut=false;toggleRecording=true;}
        else {userPaused=false;menuOpen=false;Record([&]{recorder.Event("session_resume",sequence);});Notice("");}
        return;
    }
    if(action==A::Save){if(recorder.active){recordingShortcut=false;toggleRecording=true;}return;}
    if(action==A::Reset){if(recorder.active){Notice("Сначала сохраните запись.");return;}restartAfterCalibration=false;reset=true;userPaused=true;menuOpen=true;return;}
    if(action==A::Place){if(!recorder.active&&!HumanCapture()){placeScene=true;Notice("Сцена перед вами.");}return;}
    if(action==A::ViewObserver||action==A::ViewFirstPerson){if(!HumanCapture()){firstPerson=action==A::ViewFirstPerson;refreshEgoAnchor=true;settingsDirty=true;Record([&]{recorder.Event(firstPerson?"ego_world":"observer_view",sequence);});}return;}
    if(action==A::Passthrough){if(!HumanCapture())passthroughVisible=!passthroughVisible.load();settingsDirty=true;return;}
    auto toggle=[&](std::atomic<bool>& value){value=!value.load();settingsDirty=true;refreshGeometry=true;};
    if(action==A::DebugEnabled)toggle(debugEnabled);
    if(action==A::DebugStats)toggle(debugStats);
    if(action==A::DebugMeta)toggle(debugMeta);
    if(action==A::DebugCamera)toggle(debugCamera);
    if(action==A::DebugTargets)toggle(debugTargets);
    if(action==A::DebugContacts)toggle(debugContacts);
}
void ReloadScene(){
    if(requestedScene.load()<0)return;
    std::unique_lock<std::mutex> guard(mutex,std::try_to_lock);if(!guard.owns_lock())return;
    int selected=requestedScene.exchange(-1);if(selected<0)return;
    if(recorder.active){Notice("Сначала сохраните запись.");return;}
    try{
        std::string name=questmenu::SceneName(questmenu::Scene(selected));
        auto nextSim=std::make_unique<Simulation>(assetsPath,2,name,questpolicy::VerifiedPath(assetsPath,policyCatalog.at(selectedPolicy.load())));
        auto nextVisuals=LoadVisualMeshes(name,nextSim->model);
        int nextCamera=mj_name2id(nextSim->model,mjOBJ_CAMERA,"ego");
        if(nextCamera<0)throw std::runtime_error("Missing ego camera");
        mjv_freeScene(&scene);mjv_freeScene(&workerScene);
        for(auto& entry:meshes){glDeleteBuffers(1,&entry.second.vbo);glDeleteVertexArrays(1,&entry.second.vao);}meshes.clear();
        calibrationSnapshot.reset();sim=std::move(nextSim);visualMeshes=std::move(nextVisuals);egoCamera=nextCamera;headBody=mj_name2id(sim->model,mjOBJ_BODY,"head_link");
        mjv_defaultScene(&scene);mjv_makeScene(sim->model,&scene,2048);mjv_defaultScene(&workerScene);mjv_makeScene(sim->model,&workerScene,2048);
#ifdef G1_ENABLE_FOOT_FLOOR_GUARD
        floorGuard=std::make_unique<FootFloorGuard>(sim->model,retarget->solver().model());
#endif
        retarget->calibrated=false;retarget->Pause();trackingEnabled=false;wasApplying=false;calibrationReady=false;faulted=false;
        cameraFusion.Reset();lastReference={};grip[0]=0;grip[1]=0;trackingStatus=0;pauseReason=0;calibrate=false;
        sessionScene=selected;settingsDirty=true;placeScene=true;refreshEgoAnchor=true;PublishGeometry();
        {std::lock_guard<std::mutex> statsGuard(statsMutex);stats={};stats.physicsWorkers=sim->physics_workers;}
        Notice("");__android_log_print(ANDROID_LOG_INFO,"G1Quest","Selected scene %s nq=%d nv=%d",name.c_str(),sim->model->nq,sim->model->nv);
    }catch(const std::exception& e){Notice("Не удалось загрузить сцену.");__android_log_print(ANDROID_LOG_ERROR,"G1Quest","Scene selection: %s",e.what());}
}
}

void G1Initialize(android_app* app){
    std::string assets=std::string(app->activity->internalDataPath)+"/g1";
    assetsPath=assets;
    CopyAssets(app->activity->assetManager,"",assets);
    CopyAssets(app->activity->assetManager,"meshes",assets+"/meshes");
    policyCatalog=questpolicy::Load(assets);menuState.policies=policyCatalog;
    LoadPreferences();
    std::ifstream cameraConfig(assets+"/camera_stream_url.txt");std::string cameraUrl;std::getline(cameraConfig,cameraUrl);
    cameraStream=std::make_unique<CameraStream>(cameraUrl);
    try{
        sim=std::make_unique<Simulation>(assets,2,questmenu::SceneName(questmenu::Scene(sessionScene.load())),questpolicy::VerifiedPath(assets,policyCatalog.at(selectedPolicy.load())));
    }catch(const std::exception& e){
        if(selectedPolicy==0)throw;
        __android_log_print(ANDROID_LOG_WARN,"G1Quest","Saved policy failed: %s; restoring baseline",e.what());
        selectedPolicy=0;settingsDirty=true;Notice("Политика недоступна. Загружена авторская 20k.");
        sim=std::make_unique<Simulation>(assets,2,questmenu::SceneName(questmenu::Scene(sessionScene.load())),questpolicy::VerifiedPath(assets,policyCatalog.at(0)));
    }
    retarget=std::make_unique<MetaRetargeter>(assets);
#ifdef G1_ENABLE_TWIST_GROUNDING
    retarget->EnableTwistGrounding(true);
#endif
#ifdef G1_SWING_CLEARANCE_MM
    retarget->EnableSwingClearance(G1_SWING_CLEARANCE_MM/1000.);
#endif
#ifdef G1_ENABLE_FOOT_FLOOR_GUARD
    floorGuard=std::make_unique<FootFloorGuard>(sim->model,retarget->solver().model());
#endif
    if(std::filesystem::exists(assets+"/teleop_trace.csv"))
        std::filesystem::rename(assets+"/teleop_trace.csv",assets+"/teleop_trace_previous.csv");
    trace.open(assets+"/teleop_trace.csv",std::ios::trunc);traceRows=0;
    trace<<"step,time,reset,apply_reference,grip_left,grip_right";
    for(int i=0;i<29;i++)trace<<",reference_"<<i;
    trace<<",tracking_valid,mode,ik_error,limited,height,root_qw,root_qx,root_qy,root_qz";
    for(auto name:{"head","left","right"})for(int i=0;i<7;i++)trace<<","<<name<<"_"<<i;
    trace<<"\n"<<std::setprecision(9);
    visualMeshes=LoadVisualMeshes(sim->scene_name,sim->model);
    mjv_defaultScene(&scene);mjv_makeScene(sim->model,&scene,2048);
    mjv_defaultScene(&workerScene);mjv_makeScene(sim->model,&workerScene,2048);
    egoCamera=mj_name2id(sim->model,mjOBJ_CAMERA,"ego");
    headBody=mj_name2id(sim->model,mjOBJ_BODY,"head_link");
    if(egoCamera<0)throw std::runtime_error("Missing ego camera");
    publishedGeometry.reserve(2048);PublishGeometry();
    telemetry.open(assets+"/runtime_stats.csv",std::ios::trunc);
    telemetry<<"wall_s,sim_s,cycle_ms,physics_ms,gmr_ms,inference_ms,rtf,contacts,overlap_points,pairs,max_depth_mm,solver_iterations,constraints,body_gap_ms,input_age_ms,body_confidence,body_version,state,reason,target_x,target_y,actual_x,actual_y,cmd_vx,cmd_vy,actual_vx,actual_vy,root_error,height,tilt,gmr_residual,gmr_iterations,overruns,dropped_inputs,left_foot_contacts,right_foot_contacts,leg_error_deg,physics_workers,camera_error_m,camera_error_deg,visual_scale,camera_connected,camera_state,camera_age_ms,camera_fit_mm,camera_legs,camera_weight,camera_clock_synced,camera_clock_offset_ms,camera_clock_rtt_ms,ik_position_cm,ik_orientation_deg\n";
    try{exporter=std::make_unique<RecordingExport>(app->activity->vm,app->activity->clazz);exporter->Latest(assets+"/recordings");}
    catch(const std::exception& e){__android_log_print(ANDROID_LOG_ERROR,"G1Quest","Recording export unavailable: %s",e.what());}
    running=true;
    worker=std::thread([]{
        double rateWall=ClockSeconds(),rateSim=0,lastTelemetry=0;RuntimeStats sample;sample.physicsWorkers=sim->physics_workers;
        auto next=std::chrono::steady_clock::now();
        while(running){
            double cycleStart=ClockSeconds();
            TrackingFrame input;std::chrono::steady_clock::time_point receivedAt;
            std::deque<InputSample> inputs;std::vector<std::string> events;std::deque<MenuCommand> actions;
            bool captureOnly=false,calibratedThisCycle=false;
            {std::lock_guard<std::mutex> guard(inputMutex);input=latestTracking;receivedAt=trackingTime;inputs.swap(pendingInputs);events.swap(pendingEvents);actions.swap(pendingMenuActions);}
            bool fresh=input.body.valid && (input.location_flags[0]&3)==3 &&
                std::abs(input.xr_time_ns-input.body.time_ns)<200000000LL &&
                std::chrono::steady_clock::now()-receivedAt<std::chrono::milliseconds(200);
            {
                std::lock_guard<std::mutex> guard(mutex);
                for(const auto& event:events){
                    if(event=="menu_open"||event=="user_pause")restartAfterCalibration=false;
                    if(event=="focus_pause"||event=="menu_open"||event=="user_pause"){wasApplying=false;retarget->Pause();sim->PauseWholeBodyReference();}
                    Record([&]{recorder.Event(event.c_str(),input.sequence);});
                }
                if(spaceInvalidated.exchange(false)){
                    restartAfterCalibration=false;
                    trackingEnabled=false;retarget->calibrated=false;calibrate=false;cameraFusion.Reset();
                    wasApplying=false;retarget->Pause();sim->PauseWholeBodyReference();pauseReason=1;
                    Record([&]{recorder.Event("reference_space_change",input.sequence);});
                    capture.Invalidate();
                }
                for(const auto& command:actions)ApplyMenuAction(command.action,fresh,input.sequence,command.policyIndex);
                if(reset.exchange(false)){
                    cameraFusion.Reset();pauseReason=0;sim->Reset();wasApplying=false;trackingEnabled=false;retarget->calibrated=false;
                    trackingStatus=0;calibrate=restartAfterCalibration;faulted=false;lastReference={};grip[0]=0;grip[1]=0;capture.Invalidate();refreshGeometry=true;
                    Record([&]{recorder.Event("reset",input.sequence);});Notice("");
                }
                if(retarget->calibrated&&fresh&&!retarget->Compatible(input)){
                    trackingEnabled=false;retarget->calibrated=false;capture.Invalidate();pauseReason=2;
                    Record([&]{recorder.Event("body_skeleton_changed",input.sequence);});
                }
                // Calibration must work while the menu freezes physics, also
                // after recenter in the middle of a paused recording.
                if(calibrate.load()&&fresh&&active){
                    calibrate=false;
                    try{
                        retarget->Calibrate(sim->model,sim->data,input);
                        StoreCalibration(input,std::chrono::duration_cast<std::chrono::nanoseconds>(receivedAt.time_since_epoch()).count());
                        cameraFusion.Reset();trackingEnabled=true;
                        calibratedThisCycle=true;pauseReason=0;wasApplying=false;if(capture.mode)capture.Calibrate();refreshGeometry=true;
                        Record([&]{recorder.Event("calibrate",input.sequence);});Notice("Калибровка готова.");
                        if(restartAfterCalibration){restartAfterCalibration=false;if(!menuOpen.load()){userPaused=false;Notice("Симуляция перезапущена.");}}
                    }catch(const std::exception& e){
                        restartAfterCalibration=false;
                        retarget->calibrated=false;capture.Invalidate();trackingEnabled=false;
                        Record([&]{recorder.Event("calibration_failed",input.sequence);});Notice("Калибровка не удалась. Повторите с устойчивым трекингом.");
                        __android_log_print(ANDROID_LOG_WARN,"G1Quest","Calibration: %s",e.what());
                    }
                }
                calibrationReady=retarget->calibrated;
                if(settingsDirty.exchange(false))SavePreferences();
                if(refreshGeometry.exchange(false))PublishGeometry();
                Record([&]{
                    for(const auto& entry:inputs)recorder.Input(entry.frame,entry.left,entry.right,entry.received);
                    if(calibratedThisCycle&&recorder.active){RecordCalibration();calibratedThisCycle=false;}
                    if(toggleRecording.exchange(false)){
                        bool shortcut=recordingShortcut;recordingShortcut=false;
                        if(recorder.active){
                            bool wasMenuOpen=menuOpen.load(),wasPaused=userPaused.load();
                            recorder.Event("record_button_stop",input.sequence);FinishRecording(input.sequence);
                            if(shortcut){menuOpen=wasMenuOpen;userPaused=wasPaused||faulted.load();}
                            __android_log_print(ANDROID_LOG_INFO,"G1Quest","Recording saved: %s",recorder.path().c_str());
                        }
                        else{
                            if(sessionMode!=1||!fresh||!retarget->calibrated||faulted||requestedScene>=0){
                                Notice("Для записи выберите сбор траекторий и откалибруйте позу.");return;
                            }
                            if(capture.mode){
                                firstPerson=false;retarget->EnableCameraTracking(false);capture.Start(ClockSeconds());capture.Calibrate();
                            }
                            recorder.Start(assetsPath,sim->model,questpolicy::Metadata(sim->scene_name,policyCatalog.at(selectedPolicy.load())),policyCatalog.at(selectedPolicy.load()).file,catchUpEnabled.load(),firstPerson.load());recordingStarted=ClockSeconds();acceptedRecordingSeconds=0;
                            std::ofstream config;config.exceptions(std::ios::badbit|std::ios::failbit);config.open(recorder.path()+"/session_config.json");
                            config<<"{\"schema_version\":1,\"mode\":\"trajectories\",\"scene\":\""<<sim->scene_name
                                <<"\",\"capture\":\""<<(capture.mode?"human_skeleton_only":"human_and_robot")
                                <<"\",\"plan\":\""<<(capture.mode?(capture.mode==1?"train":"test"):"manual")
                                <<"\",\"view\":\""<<(firstPerson?"first_person":"observer")<<"\",\"pause_semantics\":\"physics_frozen; pause and menu intervals excluded from demonstrations\"}\n";
                            config.close();userPaused=false;menuOpen=false;recorder.Event("record_button_start",input.sequence);
                            std::string sceneEvent="scene_"+sim->scene_name;recorder.Event(sceneEvent.c_str(),input.sequence);
                            if(capture.running){
                                capture.WritePlan(recorder.path());recorder.Event(capture.mode==1?"guide_train_start":"guide_test_start",input.sequence);
                                recorder.Event("human_skeleton_only",input.sequence);
                                captureTimeline.clear();
                                captureTimeline.exceptions(std::ios::badbit|std::ios::failbit);
                                captureTimeline.open(recorder.path()+"/capture_timeline.csv");
                                captureTimeline<<"receive_ns,sequence,stage,status,focused,source_valid,calibrated,accepted_seconds,stage_seconds\n"<<std::setprecision(17);
                            }
                            cameraRecording.clear();cameraRecording.exceptions(std::ios::badbit|std::ios::failbit);
                            cameraRecording.open(std::filesystem::path(recorder.path())/"camera.jsonl");cameraRecordedSequence=~0ull;
                            recorder.Event(firstPerson?"ego_world":"observer_view",input.sequence);
#ifdef G1_ENABLE_FOOT_FLOOR_GUARD
                            recorder.Event("floor_guard_enabled",input.sequence);
#endif
#ifdef G1_SWING_CLEARANCE_MM
                            auto swingEvent="swing_clearance_"+std::to_string(G1_SWING_CLEARANCE_MM)+"mm";
                            recorder.Event(swingEvent.c_str(),input.sequence);
#endif
#ifdef G1_ENABLE_TWIST_GROUNDING
                            recorder.Event("twist_grounding_enabled",input.sequence);
#endif
                            auto received=std::chrono::duration_cast<std::chrono::nanoseconds>(receivedAt.time_since_epoch()).count();
                            // The calibration may precede this episode. Store
                            // its exact input and robot pose before newer input.
                            recorder.Input(calibrationInput,calibrationGrip[0],calibrationGrip[1],calibrationReceived);
                            RecordCalibration();calibratedThisCycle=false;
                            if(input.sequence!=calibrationInput.sequence)recorder.Input(input,grip[0],grip[1],received);
                            __android_log_print(ANDROID_LOG_INFO,"G1Quest","Recording started: %s",recorder.path().c_str());
                        }
                    }
                    // Record optical input in every capture mode, including
                    // guided human-only capture which skips the physics loop.
                    CameraSkeleton recordedCamera;std::string recordedCameraRaw;
                    if(recorder.active && cameraStream->Latest(recordedCamera,&recordedCameraRaw) && recordedCamera.sequence!=cameraRecordedSequence){
                        cameraRecording<<"{\"input_sequence\":"<<input.sequence
                            <<",\"observed_epoch_ms\":"<<std::setprecision(17)<<CameraEpochMs()
                            <<",\"observed_monotonic_ns\":"<<EpisodeRecorder::Now()
                            <<",\"received_epoch_ms\":"<<recordedCamera.receivedMs
                            <<",\"clock_offset_ms\":"<<cameraStream->clockOffsetMs.load()
                            <<",\"clock_synced\":"<<(cameraStream->clockSynced?"true":"false")
                            <<",\"payload\":"<<recordedCameraRaw<<"}\n";
                        cameraRecordedSequence=recordedCamera.sequence;
                        static double cameraFlushed=0;double now=ClockSeconds();
                        if(now-cameraFlushed>1){cameraRecording.flush();cameraFlushed=now;}
                    }
                    if(capture.mode){
                        captureOnly=true;
                        bool resetNow=false,calibratedNow=calibratedThisCycle;
                        double now=ClockSeconds();int beforeStage=capture.stage,beforeStatus=capture.Status(active && !SessionPaused(),fresh);
                        capture.Tick(now,active && fresh && !SessionPaused());
                        int status=capture.Status(active && !SessionPaused(),fresh);
                        if(capture.running){
                            if(capture.stage!=beforeStage || status!=beforeStatus){
                                std::string event="guide_stage_"+std::to_string(capture.stage)+"_"+GuidedCapture::StatusName(status);
                                recorder.Event(event.c_str(),input.sequence);
                            }
                            captureTimeline<<EpisodeRecorder::Now()<<','<<input.sequence<<','<<capture.stage<<','<<status<<','<<active.load()<<','<<fresh<<','<<capture.calibrated<<','<<capture.Total()<<','<<capture.accepted[capture.stage]<<'\n';
                            // Frozen neutral robot state is a calibration snapshot, not a demonstration target.
                            recorder.Frame(sim->model,sim->data,sim->steps,input,resetNow,calibratedNow,false,fresh,grip[0],grip[1],0,0,false,lastReference);
                            recorder.Mimic(input,sim->steps,sim->data->time,sim->WholeBodyReference());
                            static double flushed=0;if(now-flushed>1){captureTimeline.flush();flushed=now;}
                            if(capture.completed)FinishRecording(input.sequence);
                        }
                        acceptedRecordingSeconds=capture.Total();trackingStatus=capture.calibrated?2:0;
                        {std::lock_guard<std::mutex> statsGuard(statsMutex);publishedCapture=capture;publishedCaptureFresh=fresh;}
                    }else{
                        std::lock_guard<std::mutex> statsGuard(statsMutex);publishedCapture=capture;publishedCaptureFresh=false;
                    }
                    recorder.Tick();
                });
                if(capture.running && !recorder.active){capture.Stop();captureTimeline.exceptions(std::ios::goodbit);captureTimeline.close();}
                if(recorder.active)recordedSeconds=std::max(0.,ClockSeconds()-recordingStarted.load());
                if(recorder.failed){userPaused=true;menuOpen=true;}
                captureOnly=capture.mode!=0;
            }
            if(captureOnly){std::this_thread::sleep_for(std::chrono::milliseconds(10));next=std::chrono::steady_clock::now();continue;}
            if(!active || SessionPaused() || faulted || requestedScene>=0){std::this_thread::sleep_for(std::chrono::milliseconds(10));next=std::chrono::steady_clock::now();continue;}
            try{
                std::lock_guard<std::mutex> guard(mutex);
                // The render thread can open the menu or replace a scene while
                // this worker waits for the lock. Recheck before any Step.
                if(!active||SessionPaused()||faulted||requestedScene>=0)continue;
                const bool resetNow=false;
                bool valid=input.body.valid && (input.location_flags[0]&3)==3 && std::abs(input.xr_time_ns-input.body.time_ns)<200000000LL && std::chrono::steady_clock::now()-receivedAt<std::chrono::milliseconds(200);
                bool calibratedNow=calibratedThisCycle;
                if(retarget->calibrated && valid && !retarget->Compatible(input)){
                    trackingEnabled=false;retarget->calibrated=false;pauseReason=2;
                    Record([&]{recorder.Event("body_skeleton_changed",input.sequence);});
                }
                bool applyReference=trackingEnabled && valid && retarget->calibrated;
                retarget->EnableCameraTracking(firstPerson.load());
                float leftGrip=grip[0].load(),rightGrip=grip[1].load();
                // Camera registration belongs to Quest STAGE, not to robot
                // retarget calibration. Keep diagnostics working while GMR
                // is paused or waiting for a new body bind skeleton.
                TrackingFrame fused=input;
                if(valid){
                    cameraFusion.Observe(input,CameraEpochMs());CameraSkeleton optical;
                    bool haveOptical=cameraStream->Latest(optical);
                    fused=cameraFusion.Apply(input,haveOptical?&optical:nullptr,CameraEpochMs());
                }
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
                    auto whole=retarget->Solve(fused);
                    if(catchUpEnabled){
                        auto& solver=retarget->solver();
                        const auto goal=solver.HasCameraTarget()?solver.CameraTarget():solver.CameraPose();
                        whole=ApplyCameraCatchUp(whole,goal,sim->model,sim->data,egoCamera,true);
                    }
                    sample.gmrMs=(ClockSeconds()-gmrStart)*1000;
                    double u=std::clamp((sim->data->time-blendStarted)/.5,0.,1.);
                    for(int j=0;j<35;j++)whole[j]=blendOrigin[j]+u*(whole[j]-blendOrigin[j]);
#ifdef G1_ENABLE_FOOT_FLOOR_GUARD
                    whole=floorGuard->Correct(whole);
#endif
                    sim->SetWholeBodyReference(whole);std::copy_n(whole.begin()+6,29,lastReference.begin());
                }else{sample.gmrMs=0;sim->PauseWholeBodyReference();retarget->Pause();}
                wasApplying=applyReference;
                bool poseMismatch=retarget->solver().positionRms>.10 || retarget->solver().orientationRms>.78539816339;
                trackingStatus=!valid?3:retarget->calibrated?(trackingEnabled?(valid?(poseMismatch?5:1):3):2):0;
                Record([&]{recorder.Frame(sim->model,sim->data,sim->steps,input,resetNow,calibratedNow,applyReference,valid,
                    leftGrip,rightGrip,trackingStatus,retarget->error,(poseMismatch),lastReference);});
                Record([&]{recorder.Mimic(input,sim->steps,sim->data->time,sim->WholeBodyReference());});
                // Keep the actual input episode for deterministic policy replay.
                if(trace && traceRows++<60000){
                    trace<<sim->steps<<","<<sim->data->time<<","<<resetNow<<","<<applyReference<<","<<leftGrip<<","<<rightGrip;
                    for(auto q:lastReference)trace<<","<<q;
                    trace<<","<<valid<<","<<trackingStatus.load()<<","<<retarget->error<<","<<(poseMismatch)<<","<<sim->data->qpos[2];
                    for(int i=3;i<7;i++)trace<<","<<sim->data->qpos[i];
                    for(const auto& pose:{input.head,input.hands[0],input.hands[1]}){
                        for(auto p:pose.position)trace<<","<<p;for(auto q:pose.quaternion)trace<<","<<q;
                    }
                    trace<<"\n";if(sim->steps%1000==0)trace.flush();
                }
                double physicsStart=ClockSeconds();
                for(int i=0;i<10;i++)sim->Step(false,leftGrip,rightGrip);
                if(recorder.active&&applyReference)acceptedRecordingSeconds=acceptedRecordingSeconds.load()+.01;
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
                sample.ikPositionCm=100*retarget->solver().positionRms;sample.ikOrientationDeg=retarget->solver().orientationRms*180/3.141592653589793;
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
                    telemetry<<','<<x.positionError<<','<<x.height<<','<<x.tilt<<','<<x.residual<<','<<x.gmrIterations<<','<<x.overruns<<','<<x.droppedInputs<<','<<x.footContacts[0]<<','<<x.footContacts[1]<<','<<x.legErrorDegrees<<','<<x.physicsWorkers<<','<<x.cameraPositionError<<','<<x.cameraOrientationError<<','<<x.visualScale<<','<<x.cameraConnected<<','<<x.cameraState<<','<<x.cameraAgeMs<<','<<x.cameraFitMm<<','<<x.cameraLegs<<','<<x.cameraWeight<<','<<x.cameraClockSynced<<','<<x.cameraClockOffsetMs<<','<<x.cameraClockRttMs<<','<<x.ikPositionCm<<','<<x.ikOrientationDeg<<'\n';lastTelemetry=x.published;
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
        if(interrupted){recorder.Event("app_error",latestTracking.sequence);recorder.Flush();recorder.Abort();CloseFailedRecordingStreams();capture.Stop();}
        else FinishRecording(latestTracking.sequence,"app_shutdown");
    });
    exporter.reset();cameraStream.reset();
    CloseFailedRecordingStreams();telemetry.close();trace.close();mjv_freeScene(&scene);mjv_freeScene(&workerScene);calibrationSnapshot.reset();retarget.reset();sim.reset();
}
void G1SetActive(bool value){
    if(active.exchange(value)!=value){std::lock_guard<std::mutex> guard(inputMutex);pendingEvents.emplace_back(value?"focus_resume":"focus_pause");}
}
void G1Grip(int hand,float value){if(hand>=0 && hand<2)grip[hand]=value;}
void G1Reset(){QueueMenuAction(questmenu::Action::QuickReset);}
void G1ReferenceSpaceChange(int64_t changeTime){
    std::lock_guard<std::mutex> guard(inputMutex);
    spaceChanges.Schedule(changeTime);
}
void G1SubmitTracking(const TrackingFrame& input){
    if(cameraStream)cameraStream->Submit(input);
    {std::lock_guard<std::mutex> guard(inputMutex);latestTracking=input;trackingTime=std::chrono::steady_clock::now();
        if(spaceChanges.Advance(input.xr_time_ns)){spaceInvalidated=true;placeScene=true;refreshEgoAnchor=true;anchorMenu=true;}
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
void G1Calibrate(){QueueMenuAction(questmenu::Action::Calibrate);}
void G1ToggleView(){QueueMenuAction(firstPerson?questmenu::Action::ViewObserver:questmenu::Action::ViewFirstPerson);}
void G1ToggleRecording(){
    QueueMenuAction(questmenu::Action::ToggleRecording);
}
void G1ToggleTracking(){
    if(SessionPaused()){
        if(!calibrationReady||faulted){Notice("Нужна калибровка или сброс.");return;}
        userPaused=false;menuOpen=false;
        std::lock_guard<std::mutex> guard(inputMutex);pendingEvents.emplace_back("user_resume");
    }else {userPaused=true;std::lock_guard<std::mutex> guard(inputMutex);pendingEvents.emplace_back("user_pause");}
}
void G1ToggleMenu(){
    bool open=!menuOpen.load();menuOpen=open;if(open)anchorMenu=true;
    std::lock_guard<std::mutex> guard(inputMutex);pendingEvents.emplace_back(open?"menu_open":"menu_close");
}
void G1SubmitMenuRay(int hand,const TrackedPose& pose,bool valid,bool triggerActive,float trigger){
    if(hand>=0&&hand<2)menuRays[hand]={pose,valid,triggerActive,trigger};
}
bool G1PassthroughVisible(){return HumanCapture()||passthroughVisible.load();}
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
    ReloadScene();
    if(!sim)return;
    static double last=ClockSeconds();static int frames=0;
    frames++;double now=ClockSeconds();frameNow=now;if(now-last>=.5){renderFps=frames/(now-last);frames=0;last=now;}
    {std::lock_guard<std::mutex> guard(inputMutex);frameTracking=latestTracking;menuState.notice=menuNotice;
        frameAgeMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-trackingTime).count();}
    {std::unique_lock<std::mutex> guard(statsMutex,std::try_to_lock);if(guard.owns_lock()){frameStats=stats;frameCapture=publishedCapture;frameCaptureFresh=publishedCaptureFresh;}}
    frameDrawMs=renderMs;frameStatus=trackingStatus.load();frameExportStatus=exporter?exporter->status.load():3;
    headMatrix=PoseMatrix(frameTracking.head);
    // Update optical diagnostics on the render thread even when physics is
    // paused or guided capture bypasses it. Both eyes reuse the same snapshot.
    CameraOverlay optical=frameStats.cameraOverlay;
    optical.ageMs+=std::max(0.,now-frameStats.published)*1000;
    CameraSkeleton packet;
    if(cameraStream && cameraStream->connected && cameraStream->clockSynced && cameraStream->Latest(packet)){
        if(packet.frame=="pelvis-relative"){
            CameraFusion stageMapper;
            optical=stageMapper.MapForDisplay(&packet,frameTracking,CameraEpochMs());
        }
    }else optical={};
    frameOperatorCamera=optical;
    menuState.policyIndex=selectedPolicy;menuState.catchUp=catchUpEnabled;
    menuState.mode=questmenu::Mode(sessionMode.load());menuState.scene=questmenu::Scene(sessionScene.load());
    menuState.capture=questmenu::Capture(sessionCapture.load());menuState.plan=questmenu::Plan(sessionPlan.load());
    menuState.open=menuOpen;menuState.paused=SessionPaused()||!active;menuState.recording=recordingStatus==1;
    menuState.calibrated=calibrationReady;menuState.faulted=faulted;menuState.firstPerson=firstPerson;menuState.passthrough=G1PassthroughVisible();
    menuState.trackingValid=frameTracking.body.valid&&(frameTracking.location_flags[0]&3)==3&&frameAgeMs<200&&std::abs(frameTracking.xr_time_ns-frameTracking.body.time_ns)<200000000LL;
    menuState.debugEnabled=debugEnabled;menuState.debugStats=debugStats;menuState.debugMeta=debugMeta;menuState.debugCamera=debugCamera;
    menuState.debugTargets=debugTargets;menuState.debugContacts=debugContacts;menuState.exportStatus=frameExportStatus;
    menuState.cameraInfo=!cameraStream||!cameraStream->connected?"Не подключена":!cameraStream->clockSynced?"Синхронизация часов":"Подключена";
    if(recordingStarted>0){char info[128];std::snprintf(info,sizeof(info),"Записано %.0f с   Принято %.0f с",recordedSeconds.load(),acceptedRecordingSeconds.load());menuState.recordingInfo=info;}
    static bool lastOpen=false;
    if(menuState.open&&!lastOpen){anchorMenu=true;menuTrigger={};}
    lastOpen=menuState.open;
    if(menuState.open&&(frameTracking.location_flags[0]&3)==3&&anchorMenu.exchange(false)){
        double rotation[9];mju_quat2Mat(rotation,frameTracking.head.quaternion.data());
        double fx=-rotation[2],fz=-rotation[8],length=std::hypot(fx,fz);if(length<.2){fx=0;fz=-1;length=1;}fx/=length;fz/=length;
        menuWorld={};menuWorld.m[0]=float(-fz);menuWorld.m[2]=float(fx);menuWorld.m[5]=1;
        menuWorld.m[8]=float(-fx);menuWorld.m[10]=float(-fz);menuWorld.m[15]=1;
        menuWorld.m[12]=float(frameTracking.head.position[0]+1.1*fx);menuWorld.m[13]=float(frameTracking.head.position[1]-.06);menuWorld.m[14]=float(frameTracking.head.position[2]+1.1*fz);
    }
    framePointers.clear();frameMenuRays={};auto layout=questmenu::BuildLayout(menuState);
    for(int hand=0;hand<2;hand++){
        const auto& ray=menuRays[hand];bool valid=ray.valid&&ray.active&&active;
        double rotation[9];mju_quat2Mat(rotation,ray.pose.quaternion.data());
        float origin[3],direction[3];for(int a=0;a<3;a++){origin[a]=float(ray.pose.position[a]);direction[a]=float(-rotation[a*3+2]);}
        float x=0,y=0,distance=1.4f;bool hit=valid&&menuState.open&&questmenu::RayHit(menuWorld.m,origin,direction,x,y,distance);
        auto action=hit?questmenu::HitTest(layout,x,y):questmenu::Action::None;
        framePointers.push_back({x,y,hit,action!=questmenu::Action::None});
        if(menuState.open&&valid){
            std::array<double,3> end;for(int a=0;a<3;a++)end[a]=origin[a]+(hit?distance:1.4f)*direction[a];
            frameMenuRays.bones.push_back({ray.pose.position,end,.001f,{.8f,.85f,.75f,.85f}});
        }
        if(menuTrigger[hand].Update(valid,ray.trigger>.55f)&&hit){
            using A=questmenu::Action;
            if(action==A::PageSession)menuState.page=questmenu::Page::Session;
            else if(action==A::PageView)menuState.page=questmenu::Page::View;
            else if(action==A::PageDebug)menuState.page=questmenu::Page::Debug;
            else if(action==A::PagePolicy){menuState.page=questmenu::Page::Policy;menuState.policyPage=selectedPolicy.load()/questmenu::PoliciesPerPage;}
            else if(action==A::PolicyPrevious)menuState.policyPage=std::max(0,menuState.policyPage-1);
            else if(action==A::PolicyNext)menuState.policyPage=std::min((int(policyCatalog.size())-1)/questmenu::PoliciesPerPage,menuState.policyPage+1);
            else if(action==A::Close)G1ToggleMenu();
            else if(action!=A::None)QueueMenuAction(action,questmenu::PolicyRow(action)<0?-1:menuState.policyPage*questmenu::PoliciesPerPage+questmenu::PolicyRow(action));
        }
    }

    std::unique_lock<std::mutex> guard(geometryMutex,std::try_to_lock);
    if(!guard.owns_lock()){skippedSceneUpdates++;return;}
    scene.ngeom=std::min(int(publishedGeometry.size()),scene.maxgeom);
    std::copy_n(publishedGeometry.begin(),scene.ngeom,scene.geoms);frameCamera=publishedCamera;
    frameTargets=publishedTargets;
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
    XrMatrix4x4f viewProjection,hud;std::copy_n(vp,16,viewProjection.m);XrMatrix4x4f_Multiply(&hud,&viewProjection,&headMatrix);
    auto drawInterface=[&]{
        int count=0;
        if(debugEnabled&&debugStats)count+=DrawStatsHud(hud.m,assetsPath,frameStats,renderFps,frameDrawMs,skippedSceneUpdates,frameNow,frameStatus,active.load()&&!SessionPaused(),frameExportStatus,frameTracking,frameAgeMs,firstPerson.load(),firstPerson?frameEgoScale:1.f,recordingStatus.load(),recordingStarted.load(),false);
        count+=questmenu::DrawMinimalStatus(hud.m,assetsPath,menuState,frameStatus,recordingStatus.load());
        if(menuState.open){count+=DrawWorldSkeleton(vp,frameMenuRays);count+=questmenu::DrawQuestMenu(vp,menuWorld.m,assetsPath,menuState,framePointers);}
        return count;
    };
    if(HumanCapture()){
        int rendered=DrawOperatorSkeleton(vp,frameTracking,frameAgeMs,frameOperatorCamera,frameNow,frameNow,debugEnabled&&debugMeta,debugEnabled&&debugCamera);
        if(!menuState.open)rendered+=DrawCaptureHud(hud.m,assetsPath,frameCapture,active.load()&&!SessionPaused(),frameCaptureFresh,recordingStatus.load(),frameExportStatus,frameTracking,frameAgeMs);
        rendered+=drawInterface();if(rendered>100000)throw std::runtime_error("Capture interface exceeds triangle budget");
        renderMs=.9*renderMs+.1*(ClockSeconds()-renderStart)*1000;return;
    }
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
        if(g.type==mjGEOM_PLANE&&G1PassthroughVisible())continue;
        if(g.type!=mjGEOM_MESH && g.type!=mjGEOM_BOX && g.type!=mjGEOM_PLANE && g.type!=mjGEOM_SPHERE && g.type!=mjGEOM_ELLIPSOID && g.type!=mjGEOM_CYLINDER && g.type!=mjGEOM_CAPSULE)continue;
        bool collision=debugEnabled&&debugContacts&&g.objtype==mjOBJ_GEOM&&g.objid>=0&&sim->model->geom_group[g.objid]==3;
        auto& mesh=Geometry(g,collision);
        // MuJoCo can emit thousands of contact decorations. Keep the scene
        // geometry complete and reserve the interface/operator budget; the
        // stats counter still reports every physical contact.
        if(g.category==mjCAT_DECOR&&renderedTriangles+mesh.count/3>100000-8192-2048)continue;
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
    // Use the current main renderer in STAGE metres for the operator.
    renderedTriangles+=DrawOperatorSkeleton(vp,frameTracking,frameAgeMs,frameOperatorCamera,frameNow,frameNow,debugEnabled&&debugMeta,debugEnabled&&debugCamera);
    if(debugEnabled&&debugTargets){XrMatrix4x4f targetVP;XrMatrix4x4f_Multiply(&targetVP,&viewProjection,&world);renderedTriangles+=DrawWorldSkeleton(targetVP.m,frameTargets);}
    renderedTriangles+=drawInterface();
    if(renderedTriangles>100000)throw std::runtime_error("Scene and HUD exceed triangle budget");
    renderMs=.9*renderMs+.1*(ClockSeconds()-renderStart)*1000;
}

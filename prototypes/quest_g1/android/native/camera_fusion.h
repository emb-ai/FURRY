#pragma once
#include "retarget.h"
#include <deque>
#include <string>
struct CameraJoint {std::array<double,3> p{};double confidence=0;bool measured=false;};
// Raw detector keypoint: undistorted colour-image pixels before smoothing.
struct CameraKeypoint {double u=0,v=0,confidence=0;bool valid=false;};
struct CameraSkeleton {
 // hips L/R, knees L/R, ankles L/R, nose, shoulders L/R, wrists L/R, elbows L/R
 std::array<CameraJoint,13> joints{};CameraJoint pelvis;
 uint64_t sequence=0;double sourceMs=0,receivedMs=0;std::string frame;
 // Optional 2D stream: same joint order, pinhole intrinsics fx, fy, cx, cy.
 std::array<CameraKeypoint,13> keypoints{};std::array<double,4> intrinsics{};bool hasKeypoints=false;
 // Optional PC-side AprilTag registration: p_stage = R p_camera + t.
 std::array<double,9> stageRotation{1,0,0,0,1,0,0,0,1};std::array<double,3> stageTranslation{};bool hasStagePose=false;
};
struct CameraOverlay {
 std::array<std::array<double,3>,14> points{}; // camera joints, then pelvis
 std::array<bool,14> valid{};bool aligned=false;double ageMs=-1;
};
// Time constant of each of the two frame-rate stages that smooth camera legs.
constexpr double kCameraStageSeconds=.08;
// The overlay maps raw packets: at ~15 Hz with camera-depth noise its ankles
// jumped 2 cm per packet (p90 6, p99 17 cm, 2026-10-12 17:30 at 3.9 m). Call
// once per rendered frame; the same two stages as the legs keep the overlay
// in step with what drives the robot. Points that appear or jump snap.
class CameraOverlaySmoother {
 std::array<std::array<double,3>,14> halfway{},points{};std::array<bool,14> valid{};double lastS=-1;
public:
 CameraOverlay Smooth(const CameraOverlay&,double nowS);
};
// Registration progress for the HUD. hint: 0 ok, 1 no person/shoulders,
// 2 operator side-on, 3 needs more walking, 4 inconsistent (other person?),
// 5 the stored registration stopped matching (camera moved): walk again,
// 6 2D-only packets and no AprilTag registration (press c on the PC).
// ready: legs the camera could drive now, also while legsEnabled is off.
// legs2d: legs of the last packet solved from 2D keypoints (no camera depth);
// reprojPx: their worst knee/ankle reprojection error; flipGap: the smallest
// cost margin to the mirrored knee branch (small = the view was ambiguous).
struct CameraFusionStats {int state=0,legs=0,ready=0,fitPackets=0,fitHint=1,legs2d=0;double ageMs=-1,fitMm=-1,weight=0,fitProgress=0,fitSplitDeg=-1,fitSpreadCm=0,fitInliers=0,reprojPx=-1,flipGap=-1;};
class CameraFusion {
 struct Sample{double ms;TrackingFrame frame;};
 std::deque<Sample> history;
 // One registration packet: homologous camera/STAGE landmark pairs.
 struct FitPacket{std::array<std::array<double,3>,4> from{},to{};int n=0;};
 std::deque<FitPacket> fitBuffer;uint32_t fitRandom=1;int fitCountdown=0;
 // Applied camera->STAGE registration. It depends only on where the camera
 // stands, so it outlives operator calibration and resets and is persisted.
 // One ~25 s window still errs 4-9 deg (2026-10-12 recordings), so it blends
 // the average of completed disjoint windows with the current window's fit.
 std::array<double,9> rotation{1,0,0,0,1,0,0,0,1};
 std::array<double,3> translation{};
 std::array<double,9> storedRotation{1,0,0,0,1,0,0,0,1},windowRotation{1,0,0,0,1,0,0,0,1};
 std::array<double,3> storedTranslation{},windowTranslation{};
 int storedWindows=0,windowPackets=0;bool windowFit=false;
 // Recent packets consistent with the applied registration (camera moved?).
 // A restored registration drives no legs until packets confirm it: a
 // wrong one (other camera place) can still put the pelvis within the gate.
 std::deque<char> checks;uint64_t registrationVersion=0;bool moved=false,verified=false;
 void Blend();void CommitWindow();
 // Per-packet leg target, smoothed at frame rate by two kCameraStageSeconds stages.
 std::array<std::array<double,3>,6> target{},halfway{},correction{};
 std::array<double,2> weights{},activeWeights{},lastGoodMs{};
 bool aligned=false;uint64_t lastSequence=~0ull;double lastApplyMs=0;
 // Last 2D leg solution per side, knee and ankle relative to Meta's hip:
 // the temporal depth prior and the overlay.
 std::array<std::array<double,3>,2> solvedKnee{},solvedAnkle{};std::array<double,2> solvedMs{};
 void Fit(const CameraSkeleton&,const TrackingFrame&);
public:
 CameraFusionStats stats;
 // Off: legs fade back to Meta's estimate over ~100 ms; registration keeps
 // running, so switching back to the camera needs no recalibration.
 bool legsEnabled=true;
 // Packets with 2D keypoints drive the legs by reprojection (camera depth
 // unused). Off: the depth-fused 3D path, for A/B on the same stream.
 bool use2d=true;
 // Operator-side state only (A calibration, robot reset, policy change).
 void Reset();
 // Camera moved: register from scratch.
 void ForgetRegistration();
 // The STAGE space changed (doff, Guardian, recenter): keep the stored
 // registration but drive no legs until live packets confirm it. A redone
 // Guardian fails that check and falls back to ForgetRegistration.
 void Reverify();
 // Stored registration: a text file of window count, R and t. Load applies
 // it at once; a moved camera is caught by the consistency check.
 bool LoadRegistration(const std::string& path);
 bool SaveRegistration(const std::string& path)const;
 // Bumped when the stored registration changes and should be saved.
 uint64_t RegistrationVersion()const{return registrationVersion;}
 int StoredWindows()const{return storedWindows;}
 void Observe(const TrackingFrame&,double epochMs);
 TrackingFrame Apply(const TrackingFrame&,const CameraSkeleton*,double epochMs);
 CameraOverlay MapForDisplay(const CameraSkeleton*,const TrackingFrame&,double epochMs)const;
 bool Aligned()const{return aligned;}
};

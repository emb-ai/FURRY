#include "operator_skeleton.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <cstdio>

static void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
static bool Near(const std::array<double,3>& a, const std::array<double,3>& b, double eps = 1e-6) {
    return SkeletonDistance(a, b) <= eps;
}
static bool HasBone(const OperatorSkeleton& mesh, const std::array<double,3>& a, const std::array<double,3>& b, int group) {
    for (const auto& bone : mesh.bones)
        if (bone.group == group && ((Near(bone.from, a) && Near(bone.to, b)) || (Near(bone.from, b) && Near(bone.to, a))))
            return true;
    return false;
}
static double SegmentDistance(const double* p, const std::array<double,3>& a, const std::array<double,3>& b) {
    double ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, ap[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
    double len2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
    double t = len2 < 1e-12 ? 0 : std::clamp((ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2]) / len2, 0.0, 1.0);
    double q[3] = {a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t};
    return std::sqrt((p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]) + (p[2] - q[2]) * (p[2] - q[2]));
}
static void CheckSurface(const std::vector<float>& vertices, const OperatorSkeleton& mesh, int group, float boneRadius, float jointRadius) {
    Require(vertices.size() % 9 == 0, "Triangle list is not a multiple of three vertices");
    for (size_t i = 0; i < vertices.size(); i += 3) {
        double p[3] = {vertices[i], vertices[i + 1], vertices[i + 2]};
        Require(std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]), "Non-finite skeleton vertex");
        bool explained = false;
        for (const auto& bone : mesh.bones)
            if (bone.group == group && SegmentDistance(p, bone.from, bone.to) <= boneRadius + 1e-4) explained = true;
        for (const auto& joint : mesh.joints)
            if (joint.group == group && SkeletonDistance(joint.position, std::array<double,3>{p[0], p[1], p[2]}) <= jointRadius + 1e-4) explained = true;
        Require(explained, "Vertex left its bone or joint");
    }
}

static TrackingFrame Standing() {
    TrackingFrame frame;
    frame.body.valid = true;
    frame.xr_time_ns = 1000000000;
    frame.body.time_ns = 1000000000;
    frame.location_flags[0] = 3;
    const double points[15][3] = {
        {0, 1.00, 0}, {0, 1.35, 0},
        {-0.10, 0.95, 0}, {0.10, 0.95, 0},
        {-0.10, 0.50, 0}, {0.10, 0.50, 0},
        {-0.10, 0.08, 0}, {0.10, 0.08, 0},
        {-0.20, 1.45, 0}, {0.20, 1.45, 0},
        {-0.45, 1.20, 0}, {0.45, 1.20, 0},
        {-0.55, 1.00, 0}, {0.55, 1.00, 0},
        {0, 1.70, 0}};
    for (int i = 0; i < 14; i++) {
        frame.body.flags[i] = 3;
        frame.body.joints[i].quaternion = {1, 0, 0, 0};
        for (int k = 0; k < 3; k++) frame.body.joints[i].position[k] = points[i][k];
    }
    frame.head.quaternion = {1, 0, 0, 0};
    for (int k = 0; k < 3; k++) frame.head.position[k] = points[14][k];
    return frame;
}

int main() {
    CameraOverlay none;
    auto standing = Standing();
    auto mesh = BuildOperatorSkeleton(standing, 10, none, 1, 1);
    std::array<double,3> pelvis{0, 1, 0}, chest{0, 1.35, 0}, head{0, 1.70, 0};
    std::array<double,3> hip{-0.10, 0.95, 0}, knee{-0.10, 0.50, 0}, ankle{-0.10, 0.08, 0};
    std::array<double,3> shoulder{-0.20, 1.45, 0}, elbow{-0.45, 1.20, 0}, wrist{-0.55, 1.00, 0};
    Require(HasBone(mesh, pelvis, chest, 0), "Missing pelvis-chest bone");
    Require(HasBone(mesh, shoulder, elbow, 0), "Missing upper arm");
    Require(HasBone(mesh, elbow, wrist, 0), "Missing forearm");
    Require(HasBone(mesh, hip, knee, 1), "Missing thigh");
    Require(HasBone(mesh, knee, ankle, 1), "Missing shin");
    std::array<double,3> tip = head;
    tip[1] -= kNeckClearance;
    Require(HasBone(mesh, chest, tip, 0), "Neck does not stop short of the headset");
    for (const auto& bone : mesh.bones)
        Require(!Near(bone.from, head) && !Near(bone.to, head), "Head bone reaches the headset");
    for (const auto& joint : mesh.joints)
        Require(SkeletonDistance(joint.position, head) >= kNeckClearance - 1e-6, "Joint drawn inside the headset");
    Require(mesh.camera.empty() && mesh.cameraAlpha == 0, "Absent camera overlay produced geometry");
    Require(mesh.Triangles() > 0 && mesh.Triangles() <= kOperatorSkeletonMaxTriangles, "Standing skeleton breaks the triangle cap");
    CheckSurface(mesh.upper, mesh, 0, kMetaBoneRadius, kMetaJointRadius);
    CheckSurface(mesh.legs, mesh, 1, kMetaBoneRadius, kMetaJointRadius);

    auto hidden = standing;
    hidden.body.flags[6] = 0;
    auto partial = BuildOperatorSkeleton(hidden, 10, none, 1, 1);
    Require(!HasBone(partial, knee, ankle, 1), "Invalid ankle still connected");
    for (const auto& joint : partial.joints) Require(!Near(joint.position, ankle), "Invalid ankle still drawn");

    auto distant = standing;
    distant.body.joints[6].position = {0, -8, 0};
    auto rejected = BuildOperatorSkeleton(distant, 10, none, 1, 1);
    Require(!HasBone(rejected, knee, {0, -8, 0}, 1), "Distant ankle still connected");

    auto stale = BuildOperatorSkeleton(standing, 200, none, 1, 1);
    Require(stale.bones.empty() && stale.upper.empty() && stale.legs.empty(), "Stale Meta skeleton remains");
    auto expired = standing;
    expired.body.time_ns -= 200000000LL;
    auto gap = BuildOperatorSkeleton(expired, 0, none, 1, 1);
    Require(gap.bones.empty(), "Old body sample remains");
    auto invalid = standing;
    invalid.body.valid = false;
    Require(BuildOperatorSkeleton(invalid, 0, none, 1, 1).bones.empty(), "Invalid body remains");

    CameraOverlay optical;
    constexpr int cameraPoints = int(std::tuple_size<decltype(optical.points)>::value);
    constexpr int cameraPelvis = cameraPoints - 1;
    optical.aligned = true;
    optical.ageMs = 0;
    const std::array<double,3> shift{0.30, 0, 0};
    for (int j = 0; j < 11; j++) {
        optical.valid[j] = true;
        optical.points[j] = {shift[0], 1.0, 0};
    }
    optical.valid[cameraPelvis] = true;
    optical.points[cameraPelvis] = {shift[0], 1.00, 0};
    optical.points[0] = {shift[0] - 0.10, 0.95, 0};
    optical.points[2] = {shift[0] - 0.10, 0.50, 0};
    optical.points[4] = {shift[0] - 0.10, 0.08, 0};
    optical.points[9] = {shift[0] - 0.55, 1.00, 0};
    optical.points[10] = {shift[0] + 0.55, 1.00, 0};
    auto mapped = BuildOperatorSkeleton(standing, 10, optical, 1, 1);
    Require(std::abs(mapped.cameraAlpha - 1.f) < 1e-6, "Fresh camera overlay is not opaque");
    Require(HasBone(mapped, optical.points[0], optical.points[2], 2), "Camera thigh missing");
    Require(HasBone(mapped, optical.points[2], optical.points[4], 2), "Camera shin missing");
    Require(!HasBone(mapped, optical.points[7], optical.points[9], 2), "Camera invented an elbow");
    bool wristMarker = false;
    for (const auto& joint : mapped.joints)
        if (joint.group == 2 && Near(joint.position, optical.points[9])) wristMarker = true;
    Require(wristMarker, "Measured wrist marker missing");
    CheckSurface(mapped.camera, mapped, 2, kCameraBoneRadius, kCameraJointRadius);
    Require(mapped.Triangles() <= kOperatorSkeletonMaxTriangles, "Mapped skeleton breaks the triangle cap");
    for (const auto& bone : mapped.bones)
        if (bone.group < 2) Require(HasBone(mesh, bone.from, bone.to, bone.group), "Camera shift moved the Meta skeleton");

    auto bothHidden = BuildOperatorSkeleton(standing, 10, optical, 1, 1, false, false);
    Require(bothHidden.bones.empty() && bothHidden.joints.empty() && bothHidden.Triangles() == 0 &&
            bothHidden.cameraAlpha == 0, "Disabled overlays produced geometry");
    auto metaOnly = BuildOperatorSkeleton(standing, 10, optical, 1, 1, true, false);
    Require(metaOnly.upper == mesh.upper && metaOnly.legs == mesh.legs && metaOnly.camera.empty() &&
            metaOnly.cameraAlpha == 0, "Meta-only toggle changes Meta or retains camera geometry");
    Require(std::all_of(metaOnly.joints.begin(), metaOnly.joints.end(), [](const auto& joint) { return joint.group < 2; }),
            "Meta-only toggle retains camera joints");
    auto cameraOnly = BuildOperatorSkeleton(standing, 10, optical, 1, 1, false, true);
    Require(cameraOnly.upper.empty() && cameraOnly.legs.empty() && cameraOnly.camera == mapped.camera &&
            cameraOnly.cameraAlpha == mapped.cameraAlpha, "Camera-only toggle changes placement or retains Meta geometry");
    Require(std::all_of(cameraOnly.bones.begin(), cameraOnly.bones.end(), [](const auto& bone) { return bone.group == 2; }) &&
            std::all_of(cameraOnly.joints.begin(), cameraOnly.joints.end(), [](const auto& joint) { return joint.group == 2; }),
            "Camera-only toggle retains Meta bones or joints");

    // Hiding Meta must not switch the reach gate to the displaced camera's own
    // pelvis, which would otherwise make a gross alignment error look valid.
    auto displacedCamera = optical;
    for (auto& p : displacedCamera.points) p[0] += 10;
    auto gatedCamera = BuildOperatorSkeleton(standing, 10, displacedCamera, 1, 1, false, true);
    Require(gatedCamera.camera.empty() && gatedCamera.joints.empty() && gatedCamera.bones.empty(),
            "Camera reach gate vanished with the Meta overlay");
    auto absentMeta = standing;
    absentMeta.body.valid = false;
    auto independentCamera = BuildOperatorSkeleton(absentMeta, 10, displacedCamera, 1, 1, false, true);
    Require(!independentCamera.camera.empty(), "Independent camera lost its pelvis reach anchor without Meta tracking");

    if constexpr (cameraPoints >= 14) {
        auto measuredArms = optical;
        measuredArms.valid[11] = measuredArms.valid[12] = true;
        measuredArms.points[7] = {shift[0] - .20, 1.45, 0};
        measuredArms.points[8] = {shift[0] + .20, 1.45, 0};
        measuredArms.points[11] = {shift[0] - .45, 1.20, 0};
        measuredArms.points[12] = {shift[0] + .45, 1.20, 0};
        auto armMesh = BuildOperatorSkeleton(standing, 10, measuredArms, 1, 1, false, true);
        Require(HasBone(armMesh, measuredArms.points[7], measuredArms.points[11], 2) &&
                HasBone(armMesh, measuredArms.points[11], measuredArms.points[9], 2) &&
                HasBone(armMesh, measuredArms.points[8], measuredArms.points[12], 2) &&
                HasBone(armMesh, measuredArms.points[12], measuredArms.points[10], 2),
                "Measured camera elbows are not connected to shoulders and wrists");
        Require(!HasBone(armMesh, measuredArms.points[7], measuredArms.points[9], 2),
                "Camera arm bypasses a measured elbow");
        measuredArms.valid[11] = false;
        auto noElbow = BuildOperatorSkeleton(standing, 10, measuredArms, 1, 1, false, true);
        Require(!HasBone(noElbow, measuredArms.points[7], measuredArms.points[11], 2) &&
                !HasBone(noElbow, measuredArms.points[11], measuredArms.points[9], 2),
                "Camera draws an invalid elbow");
    }

    optical.points[4] = {shift[0], -8, 0};
    auto farCamera = BuildOperatorSkeleton(standing, 10, optical, 1, 1);
    Require(!HasBone(farCamera, optical.points[2], optical.points[4], 2), "Distant camera ankle still connected");

    optical.points[4] = {shift[0] - 0.10, 0.08, 0};
    optical.ageMs = 200;
    auto fading = BuildOperatorSkeleton(standing, 10, optical, 1, 1);
    Require(std::abs(fading.cameraAlpha - float((500.0 - 200.0) / 350.0)) < 1e-5, "Camera fade does not match the inset");
    optical.aligned = false;
    Require(BuildOperatorSkeleton(standing, 10, optical, 1, 1).camera.empty(), "Uncalibrated camera overlay drawn");
    optical.aligned = true;
    optical.ageMs = 0;
    auto stalled = BuildOperatorSkeleton(standing, 10, optical, 2, 1);
    Require(stalled.camera.empty() && stalled.bones.size() == mesh.bones.size(), "Stalled publish kept the camera skeleton");
    optical.ageMs = 500;
    Require(BuildOperatorSkeleton(standing, 10, optical, 1, 1).camera.empty(), "Expired camera overlay drawn");

    puts("Operator skeleton placement, expiry, independent visibility, camera reach gate and triangle cap checks passed");
}

#pragma once
#include "camera_fusion.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>
#include <vector>

// World-space skeleton on the operator. Positions stay in STAGE metres and are
// drawn with the eye view, not the robot scene transform. Meta upper body is
// cyan, estimated legs are orange, and measured camera landmarks are a larger
// pink shell so a registration error remains visible beside the Meta bone.
// The neck stops short of the headset: the head pose is the HMD itself.
inline constexpr float kMetaBoneRadius = 0.012f;
inline constexpr float kMetaJointRadius = 0.020f;
inline constexpr float kCameraBoneRadius = 0.018f;
inline constexpr float kCameraJointRadius = 0.028f;
inline constexpr double kNeckClearance = 0.18;
inline constexpr double kOperatorReach = 2.5;
inline constexpr int kOperatorSkeletonMaxTriangles = 1024;

struct OverlayBone {
    std::array<double,3> from{}, to{};
    int group = 0; // 0 upper, 1 legs, 2 camera
};
struct OverlayJoint {
    std::array<double,3> position{};
    int group = 0;
};
struct OperatorSkeleton {
    std::vector<OverlayBone> bones;
    std::vector<OverlayJoint> joints;
    std::vector<float> upper, legs, camera;
    float cameraAlpha = 0;
    int Triangles() const { return int((upper.size() + legs.size() + camera.size()) / 9); }
};

inline bool SkeletonFinite(const std::array<double,3>& p) {
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}
inline double SkeletonDistance(const std::array<double,3>& a, const std::array<double,3>& b) {
    double d = 0;
    for (int i = 0; i < 3; i++) d += (a[i] - b[i]) * (a[i] - b[i]);
    return std::sqrt(d);
}

inline void AppendTriangle(std::vector<float>& out, const double* a, const double* b, const double* c) {
    const double* p[] = {a, b, c};
    for (const double* v : p) for (int i = 0; i < 3; i++) out.push_back(float(v[i]));
}
inline void AppendOctahedron(std::vector<float>& out, const std::array<double,3>& c, float radius) {
    double p[6][3] = {
        {c[0] + radius, c[1], c[2]}, {c[0] - radius, c[1], c[2]},
        {c[0], c[1] + radius, c[2]}, {c[0], c[1] - radius, c[2]},
        {c[0], c[1], c[2] + radius}, {c[0], c[1], c[2] - radius}};
    const int faces[8][3] = {{0,2,4},{0,4,3},{0,3,5},{0,5,2},{1,4,2},{1,3,4},{1,5,3},{1,2,5}};
    for (auto& f : faces) AppendTriangle(out, p[f[0]], p[f[1]], p[f[2]]);
}
inline void AppendCylinder(std::vector<float>& out, const std::array<double,3>& a, const std::array<double,3>& b, float radius) {
    double d[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    double len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    if (!(len > 1e-4)) return;
    double dx = d[0] / len, dy = d[1] / len, dz = d[2] / len;
    double ax = 0, ay = 0, az = 1;
    if (std::fabs(dz) > 0.9) { ax = 1; az = 0; }
    double ux = dy * az - dz * ay, uy = dz * ax - dx * az, uz = dx * ay - dy * ax;
    double ul = std::sqrt(ux * ux + uy * uy + uz * uz);
    if (!(ul > 1e-8)) return;
    ux /= ul; uy /= ul; uz /= ul;
    double vx = dy * uz - dz * uy, vy = dz * ux - dx * uz, vz = dx * uy - dy * ux;
    double u[3] = {ux, uy, uz}, v[3] = {vx, vy, vz};
    const int sides = 6;
    for (int i = 0; i < sides; i++) {
        double t0 = i * 6.283185307179586 / sides, t1 = (i + 1) * 6.283185307179586 / sides;
        double c0 = std::cos(t0), s0 = std::sin(t0), c1 = std::cos(t1), s1 = std::sin(t1);
        double p0[3], p1[3], q0[3], q1[3];
        for (int k = 0; k < 3; k++) {
            double n0 = radius * (u[k] * c0 + v[k] * s0), n1 = radius * (u[k] * c1 + v[k] * s1);
            p0[k] = a[k] + n0; p1[k] = a[k] + n1; q0[k] = b[k] + n0; q1[k] = b[k] + n1;
        }
        AppendTriangle(out, p0, q0, p1);
        AppendTriangle(out, p1, q0, q1);
    }
}

inline OperatorSkeleton BuildOperatorSkeleton(const TrackingFrame& raw, double rawAgeMs, const CameraOverlay& optical, double now, double published,
                                               bool showMeta = true, bool showCamera = true) {
    OperatorSkeleton mesh;
    if (!showMeta && !showCamera) return mesh;
    auto buffer = [&](int group) -> std::vector<float>& {
        return group == 0 ? mesh.upper : group == 1 ? mesh.legs : mesh.camera;
    };
    auto addJoint = [&](int group, const std::array<double,3>& p, float radius) {
        mesh.joints.push_back({p, group});
        AppendOctahedron(buffer(group), p, radius);
    };
    auto addBone = [&](int group, const std::array<double,3>& a, const std::array<double,3>& b, float radius) {
        double length = SkeletonDistance(a, b);
        if (length < 0.02 || length > 1.5) return;
        mesh.bones.push_back({a, b, group});
        AppendCylinder(buffer(group), a, b, radius);
    };
    bool fresh = raw.body.valid && rawAgeMs >= 0 && rawAgeMs < 200 &&
        std::abs(raw.xr_time_ns - raw.body.time_ns) < 200000000LL;
    std::array<std::array<double,3>,15> meta{};
    std::array<bool,15> metaOk{};
    bool haveAnchor = false;
    std::array<double,3> anchor{};
    if (fresh) {
        for (int j = 0; j < 15; j++) {
            const auto& p = j < 14 ? raw.body.joints[j].position : raw.head.position;
            auto flags = j < 14 ? raw.body.flags[j] : raw.location_flags[0];
            if ((flags & 3) == 3 && SkeletonFinite(p)) { meta[j] = p; metaOk[j] = true; }
        }
        if (metaOk[0]) { anchor = meta[0]; haveAnchor = true; }
        else if (metaOk[14]) { anchor = meta[14]; haveAnchor = true; }
        if (!haveAnchor) metaOk.fill(false);
        for (int j = 0; j < 15; j++) if (metaOk[j] && SkeletonDistance(meta[j], anchor) > kOperatorReach) metaOk[j] = false;
        // The hidden Meta skeleton still supplies the physical reach anchor
        // used to reject distant camera points. Visibility changes geometry.
        if (showMeta) {
            const std::pair<int,int> upperEdges[] = {{0,1},{1,8},{8,10},{10,12},{1,9},{9,11},{11,13}};
            const std::pair<int,int> legEdges[] = {{0,2},{2,4},{4,6},{0,3},{3,5},{5,7}};
            for (auto e : upperEdges) if (metaOk[e.first] && metaOk[e.second]) addBone(0, meta[e.first], meta[e.second], kMetaBoneRadius);
            for (auto e : legEdges) if (metaOk[e.first] && metaOk[e.second]) addBone(1, meta[e.first], meta[e.second], kMetaBoneRadius);
            if (metaOk[1] && metaOk[14]) {
                auto chest = meta[1], head = meta[14];
                double length = SkeletonDistance(chest, head);
                if (length > kNeckClearance + 0.04 && length <= 1.5) {
                    std::array<double,3> tip{};
                    for (int k = 0; k < 3; k++) tip[k] = head[k] - (head[k] - chest[k]) * (kNeckClearance / length);
                    addBone(0, chest, tip, kMetaBoneRadius);
                    addJoint(0, tip, kMetaJointRadius);
                }
            }
            for (int j = 0; j < 14; j++) if (metaOk[j]) addJoint(j >= 2 && j <= 7 ? 1 : 0, meta[j], kMetaJointRadius);
        }
    }
    double opticalAge = optical.ageMs + std::max(0.0, now - published) * 1000.0;
    if (showCamera && optical.aligned && optical.ageMs >= 0 && opticalAge < 500) {
        constexpr size_t cameraCount = std::tuple_size<decltype(optical.points)>::value;
        constexpr int pelvisIndex = int(cameraCount) - 1;
        mesh.cameraAlpha = float(std::clamp((500.0 - opticalAge) / 350.0, 0.0, 1.0));
        std::array<double,3> camAnchor = anchor;
        bool gate = haveAnchor;
        if (!gate && optical.valid[pelvisIndex] && SkeletonFinite(optical.points[pelvisIndex])) {
            camAnchor = optical.points[pelvisIndex]; gate = true;
        }
        std::array<std::array<double,3>,cameraCount> pts{};
        std::array<bool,cameraCount> ok{};
        for (size_t j = 0; j < cameraCount; j++) {
            if (!optical.valid[j] || !SkeletonFinite(optical.points[j])) continue;
            if (gate && SkeletonDistance(optical.points[j], camAnchor) > kOperatorReach) continue;
            pts[j] = optical.points[j];
            ok[j] = true;
        }
        // Pelvis is the last overlay point in both legacy and current packets.
        // Never shortcut shoulders to wrists when no measured elbow exists.
        const std::pair<int,int> edges[] = {{0,2},{2,4},{1,3},{3,5},{0,1},
            {pelvisIndex,0},{pelvisIndex,1},{7,8},{7,0},{8,1},{6,7},{6,8}};
        for (auto e : edges) if (ok[e.first] && ok[e.second]) addBone(2, pts[e.first], pts[e.second], kCameraBoneRadius);
        if constexpr (cameraCount >= 14) {
            const std::pair<int,int> arms[] = {{7,11},{11,9},{8,12},{12,10}};
            for (auto e : arms) if (ok[e.first] && ok[e.second]) addBone(2, pts[e.first], pts[e.second], kCameraBoneRadius);
        }
        for (size_t j = 0; j < cameraCount; j++) if (ok[j]) addJoint(2, pts[j], kCameraJointRadius);
    }
    return mesh;
}

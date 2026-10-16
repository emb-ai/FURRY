#pragma once
#include <algorithm>
#include <cmath>
#include <openxr/openxr.h>

namespace questcapture {
// Quest's flat MR capture may sample the nasal region outside the native
// asymmetric eye frustum. Supply symmetric coverage during operator capture.
// 57 degrees covers the configured 110-degree capture with 2 degrees of margin.
// The same result MUST be used for rendering and XrCompositionLayerProjectionView.
inline XrFovf ProjectionFov(const XrFovf& native, bool recording) {
    if (!recording) return native;
    constexpr float minimumHalfFov = 57.f * 3.14159265358979323846f / 180.f;
    const float horizontal = std::max({minimumHalfFov, std::abs(native.angleLeft), std::abs(native.angleRight)});
    const float vertical = std::max({minimumHalfFov, std::abs(native.angleUp), std::abs(native.angleDown)});
    return {-horizontal, horizontal, vertical, -vertical};
}
}

#pragma once
#include <array>
#include <mujoco/mujoco.h>

// mjvGeom already expands cylinder/capsule MJCF (radius, half-height) into
// render XYZ (radius, radius, half-height). Do not apply that conversion twice.
inline std::array<float, 3> RenderGeomSize(const mjvGeom &g) {
  if (g.type == mjGEOM_MESH)
    return {1, 1, 1};
  if (g.type == mjGEOM_PLANE)
    return {15, 15, .005f};
  return {g.size[0], g.size[1], g.size[2]};
}

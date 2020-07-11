#pragma once

#include <cfloat>
#include <unordered_set>
#include <math/Types.h>
#include <math/Transform3f.h>
#include <geometry/GeometryData.h>

struct SurfaceDistant : public GeometryData
{
    SurfaceDistant(const float& u, const float& v) : u(u), v(v), opacity(VEC3F_ONES) {}

    float u, v;

    Vec3f opacity;

    // Material * material

};

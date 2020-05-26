#pragma once

#include <cfloat>
#include <unordered_set>
#include <Math/Types.h>
#include <Math/Transform3f.h>
#include <geometry/GeometryData.h>

struct GeometryDistant : public GeometryData
{
    GeometryDistant(const float& u, const float& v, const Vec3f& wi) : opacity(VEC3F_ONES), u(u), v(v), wi(wi) {}

    Vec3f opacity;
    void set_opacity(const Vec3f& _opacity) { opacity = _opacity; }

    float u, v;
    Vec3f wi;
};

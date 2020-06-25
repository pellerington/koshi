#pragma once

#include <cfloat>
#include <unordered_set>
#include <Math/Types.h>
#include <Math/Transform3f.h>
#include <geometry/GeometryData.h>

struct Surface : public GeometryData
{
    Surface(const Vec3f& position, const Vec3f& normal, const float& u, const float& v, const float& w, const bool& facing)
    : opacity(VEC3F_ONES), position(position), normal(normal), facing(facing), u(u), v(v), w(w)
    {
        front_position = position + normal *  RAY_OFFSET;
        back_position  = position + normal * -RAY_OFFSET;
    }

    Vec3f opacity;
    void set_opacity(const Vec3f& _opacity) { opacity = _opacity; }

    Vec3f position;
    Vec3f front_position;
    Vec3f back_position;

    Vec3f normal;
    bool facing;

    float u, v, w;
};

#pragma once

#include <cfloat>
#include <unordered_set>
#include <Math/Types.h>
#include <Math/Transform3f.h>
#include <geometry/GeometryData.h>

struct GeometrySurface : public GeometryData
{
    GeometrySurface(const Vec3f& position, const Vec3f& normal, const float& u, const float& v, const Vec3f& wi = VEC3F_ZERO)
    : opacity(VEC3F_ONES), position(position), normal(normal), u(u), v(v), wi(wi)
    {
        front_position = position + normal *  RAY_OFFSET;
        back_position  = position + normal * -RAY_OFFSET;
        transform = Transform3f::basis_transform(normal);
        n_dot_wi = normal.dot(-wi);
        facing = (n_dot_wi >= 0.f);
    }

    Vec3f opacity;
    void set_opacity(const Vec3f& _opacity) { opacity = _opacity; }

    Vec3f position;
    Vec3f front_position;
    Vec3f back_position;

    Vec3f normal;
    Transform3f transform;
    bool facing;

    float u, v;

    Vec3f wi;
    float n_dot_wi;
};

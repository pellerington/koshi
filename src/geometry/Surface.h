#pragma once

#include <cfloat>
#include <unordered_set>
#include <math/Types.h>
#include <math/Transform3f.h>
#include <geometry/GeometryData.h>

class Material;

struct Surface : public GeometryData
{
    Surface(const Vec3f& position, const Vec3f& normal, const float& u, const float& v, const float& w, const bool& facing)
    : position(position), normal(normal), facing(facing), u(u), v(v), w(w), opacity(VEC3F_ONES), material(nullptr)
    {
        front_position = position + normal *  RAY_OFFSET;
        back_position  = position + normal * -RAY_OFFSET;
    }

    Vec3f position;
    Vec3f front_position;
    Vec3f back_position;

    Vec3f normal;
    bool facing;

    float u, v, w;

    Vec3f opacity;

    Material * material;
};

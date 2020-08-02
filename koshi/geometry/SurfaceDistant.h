#pragma once

#include <koshi/math/Types.h>
#include <koshi/geometry/GeometryData.h>

class Material;

struct SurfaceDistant : public GeometryData
{
    SurfaceDistant(const float& u, const float& v, const float& w) 
    : u(u), v(v), opacity(VEC3F_ONES), material(nullptr) 
    {
    }

    float u, v, w;

    Vec3f opacity;

    Material * material;
};

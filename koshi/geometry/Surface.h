#pragma once

#include <cfloat>
#include <unordered_set>
#include <koshi/math/Types.h>
#include <koshi/math/Transform3f.h>
#include <koshi/geometry/SurfaceDistant.h>

struct Surface : public SurfaceDistant
{
    Surface(const Vec3f& position, const Vec3f& normal, const float& u, const float& v, const float& w, const bool& facing)
    : SurfaceDistant(u, v, w), position(position), normal(normal), facing(facing)
    {
        front_position = position + normal *  RAY_OFFSET;
        back_position  = position + normal * -RAY_OFFSET;
    }

    Vec3f position;
    Vec3f front_position;
    Vec3f back_position;

    Vec3f normal;
    bool facing;
};

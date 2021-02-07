#pragma once

#include <koshi/math/Vec3.h>
#include <koshi/math/Transform.h>

KOSHI_OPEN_NAMESPACE

class Geometry;

struct Intersect
{
    Geometry * geometry;
    uint prim;
    Transform obj_to_world;
    // Material * material / Integrator * integrator;
    Vec3f position;
    Vec3f normal;
    bool facing;
    float t, t_max;
    Vec3f uvw, uvw_max;
};

KOSHI_CLOSE_NAMESPACE
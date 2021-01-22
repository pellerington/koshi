#pragma once

#include <koshi/Vec3.h>

KOSHI_OPEN_NAMESPACE

class Geometry;

struct Intersect
{
    Geometry * geometry;
    uint prim;
    // World to Object ?
    // Material * material / Integrator * integrator;
    Vec3f position;
    Vec3f normal;
    float t, t_max;
    Vec3f uvw, uvw_max;
};

KOSHI_CLOSE_NAMESPACE
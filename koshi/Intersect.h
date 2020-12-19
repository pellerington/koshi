#pragma once

#include <koshi/Vec3.h>

KOSHI_OPEN_NAMESPACE

class Geometry;

struct Intersect
{
    Geometry * geometry;
    uint primId;
    // World to Object ?
    // Material * material / Integrator * integrator;
    Vec3f position;
    Vec3f normal;
    float t0, t1;
    Vec3f uvw0;
    Vec3f uvw1;
};

KOSHI_CLOSE_NAMESPACE
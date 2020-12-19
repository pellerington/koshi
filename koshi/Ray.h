#pragma once

#include <koshi/Vec3.h>

KOSHI_OPEN_NAMESPACE

struct Ray
{
    Vec3f origin;
    Vec3f direction;
    float tmin, tmax;
};

KOSHI_CLOSE_NAMESPACE
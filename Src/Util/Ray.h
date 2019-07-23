#pragma once

#include <cfloat>
#include "../Math/Types.h"

struct Ray
{
    Vec3f o;
    Vec3f dir;
    float t = FLT_MAX;
    bool hit = false;
    uint depth = 0;
    Vec3f inv_dir;
};

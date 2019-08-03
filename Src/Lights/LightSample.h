#pragma once

#include "../Math/Types.h"

struct LightSample {
    Vec3f position;
    Vec3f intensity;
    float pdf = 0.f;
    float t = 0.f;
    uint id;
};

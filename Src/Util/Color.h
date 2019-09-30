#pragma once

#include "../Math/Types.h"

inline bool is_black(const Vec3f& color)
{
    return !color.r && !color.g && !color.b;
}

inline bool is_saturated(const Vec3f &color)
{
    return color.r >= 1.f && color.g >= 1.f && color.b >= 1.f;
}

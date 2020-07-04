#pragma once

#include <math/Types.h>

inline bool is_black(const Vec3f& color)
{
    return !color.r && !color.g && !color.b;
}

inline bool is_saturated(const Vec3f& color)
{
    return color.r >= 1.f && color.g >= 1.f && color.b >= 1.f;
}

inline float luminance(const Vec3f& color)
{
    return (0.2126f*color.r + 0.7152f*color.g + 0.0722f*color.b);
}
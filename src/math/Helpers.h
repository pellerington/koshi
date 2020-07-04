#pragma once

#include <intersection/Ray.h>

inline float clamp(const float& value, const float& min, const float& max)
{
    return (value < min) ? min : ((value > max) ? max : value);
}

inline float variance(const Vec3f& sum, const Vec3f& sum_sqr, const float& num_samples)
{
    return ((sum_sqr - ((sum * sum) / num_samples)) / (num_samples - 1.f)).max();
}
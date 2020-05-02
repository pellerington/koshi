#pragma once

#include <intersection/Ray.h>

inline float clamp(const float &value, const float &min, const float &max)
{
    return (value < min) ? min : ((value > max) ? max : value);
}
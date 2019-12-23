#pragma once

#include "Texture.h"

class Gradient : public Texture
{
public:
    Gradient(const Vec3f &min = VEC3F_ZERO, const Vec3f &max = VEC3F_ONES, const uint axis = 0.f)
    : min(min), max(max), axis(axis) {}

    const Vec3f get_vec3f(const float &u, const float &v, const float &w)
    {
        const float weight = (axis == 0) ? u - floor(u) : (axis == 1) ? v - floor(v) : w - floor(w);
        return weight * max + (1.f - weight) * min;
    }

    const float get_float(const float &u, const float &v, const float &w)
    {
        return get_vec3f(u, v, w)[0];
    }

private:
    const Vec3f min, max;
    const uint axis;
};

#pragma once

#include "Texture.h"

class Gradient : public Texture
{
public:
    Gradient(const Vec3f &min = VEC3F_ZERO, const Vec3f &max = VEC3F_ONES, const uint axis = 0.f)
    : min(min), max(max), axis(axis) {}

    const bool get_vec3f(const float &u, const float &v, const float &w, Vec3f &out)
    {
        const float weight = (axis == 0) ? u - floor(u) : (axis == 1) ? v - floor(v) : w - floor(w);
        out = weight * max + (1.f - weight) * min;
        return true;
    }

    const bool get_float(const float &u, const float &v, const float &w, float &out)
    {
        Vec3f vec_out;
        if(get_vec3f(u, v, w, vec_out))
        {
            out = vec_out[0];
            return true;
        }
        return false;
    }

private:
    const Vec3f min, max;
    const uint axis;
};

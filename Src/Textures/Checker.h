#pragma once

#include "Texture.h"

class Checker : public Texture
{
public:
    Checker(const Vec3f scale = VEC3F_ONES)
    : scale(scale) {}

    const bool get_vec3f(const float &u, const float &v, const float &w, Vec3f &out)
    {
        int i = 1;
        Vec3f uvw = scale * Vec3f(u, v, w);
        uvw = uvw - Vec3f::floor(uvw);
        i *= (uvw.u > 0.5f) ? -1 : 1;
        i *= (uvw.v > 0.5f) ? -1 : 1;
        i *= (uvw.w > 0.5f) ? -1 : 1;
        out = (i > 0) ? VEC3F_ONES : VEC3F_ZERO;
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
    const Vec3f scale;
};

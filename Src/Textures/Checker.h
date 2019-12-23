#pragma once

#include "Texture.h"

class Checker : public Texture
{
public:
    Checker(const Vec3f scale = VEC3F_ONES)
    : scale(scale) {}

    const Vec3f get_vec3f(const float &u, const float &v, const float &w)
    {
        int i = 1;
        Vec3f uvw = scale * Vec3f(u, v, w);
        uvw = uvw - Vec3f::floor(uvw);
        i *= (uvw.u > 0.5f) ? -1 : 1;
        i *= (uvw.v > 0.5f) ? -1 : 1;
        i *= (uvw.w > 0.5f) ? -1 : 1;
        return (i > 0) ? VEC3F_ONES : VEC3F_ZERO;
    }

    const float get_float(const float &u, const float &v, const float &w)
    {
        return get_vec3f(u, v, w)[0];
    }

private:
    const Vec3f scale;
};

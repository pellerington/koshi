#pragma once

#include <Textures/Texture.h>

class TextureGradient : public Texture
{
public:
    TextureGradient(const Vec3f &min = VEC3F_ZERO, const Vec3f &max = VEC3F_ONES, const uint axis = 0.f)
    : min(min), max(max), axis(axis) {}

    Vec3f get_vec3f(const float &u, const float &v, const float &w, Resources &resources)
    {
        const float weight = (axis == 0) ? u - floor(u) : (axis == 1) ? v - floor(v) : w - floor(w);
        return weight * max + (1.f - weight) * min;
    }

    float get_float(const float &u, const float &v, const float &w, Resources &resources)
    {
        return get_vec3f(u, v, w, resources)[0];
    }

private:
    const Vec3f min, max;
    const uint axis;
};

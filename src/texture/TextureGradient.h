#pragma once

#include <texture/Texture.h>

class TextureGradient : public Texture
{
public:
    TextureGradient(const Vec3f& min = VEC3F_ZERO, const Vec3f& max = VEC3F_ONES, const uint& axis = 0)
    : min(min), max(max), axis(axis) {}

    Vec3f evaluate(const float& u, const float& v, const float& w, const Intersect * intersect, Resources& resources) const
    {
        const float weight = (axis == 0) ? u - floor(u) : (axis == 1) ? v - floor(v) : w - floor(w);
        return weight * max + (1.f - weight) * min;
    }

    Vec3f delta() const { return VEC3F_ZERO; }

    bool null() const { return min.null() && max.null(); }

private:
    const Vec3f min, max;
    const uint axis;
};

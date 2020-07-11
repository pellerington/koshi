#pragma once

#include <texture/Texture.h>

class TextureChecker : public Texture
{
public:
    TextureChecker(const Vec3f scale = VEC3F_ONES)
    : scale(scale) 
    {
    }

    Vec3f evaluate(const float& u, const float& v, const float& w, const Intersect * intersect, Resources& resources) const
    {
        int i = 1;
        Vec3f uvw = scale * Vec3f(u, v, w);
        uvw = uvw - Vec3f::floor(uvw);
        i *= (uvw.u > 0.5f) ? -1 : 1;
        i *= (uvw.v > 0.5f) ? -1 : 1;
        i *= (uvw.w > 0.5f) ? -1 : 1;
        return (i > 0) ? VEC3F_ONES : VEC3F_ZERO;
    }

    virtual Vec3f delta() const { return 1.f / (scale * Vec3f(2.f, 2.f, 2.f)); };

    virtual bool null() const { return false; };

private:
    const Vec3f scale;
};

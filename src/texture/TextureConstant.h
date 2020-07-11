#pragma once

#include <texture/Texture.h>
#include <math/Types.h>

class TextureConstant : public Texture
{
public:
    TextureConstant(const Vec3f& color)
    : color(color) 
    {
    }

    Vec3f evaluate(const float& u, const float& v, const float& w, const Intersect * intersect, Resources& resources) const
    {
        return color;
    }

    Vec3f delta() const { return VEC3F_ONES; }

    bool null() const { return color.null(); }

private:
    const Vec3f color;
};
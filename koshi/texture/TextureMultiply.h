#pragma once

#include <koshi/texture/Texture.h>
#include <koshi/math/Types.h>

class TextureMultiply : public Texture
{
public:
    TextureMultiply(const Vec3f& value, const Texture * texture = nullptr)
    : value(value), texture(texture) 
    {
    }

    Vec3f evaluate(const float& u, const float& v, const float& w, const Intersect * intersect, Resources& resources) const
    {
        if(texture)
            return value * texture->evaluate(u, v, w, intersect, resources);
        return value;
    }

    Vec3f delta() const { return !texture ? VEC3F_ONES : texture->delta(); }

    bool null() const { return value.null() && !texture; }

private:
    const Vec3f value;
    const Texture * texture;
};
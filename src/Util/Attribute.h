#pragma once

#include <Textures/Texture.h>
#include <Math/Types.h>

class AttributeVec3f
{
public:
    AttributeVec3f(const Vec3f &value)
    : value(value), texture(nullptr) {}
    AttributeVec3f(Texture * texture, const Vec3f &gain = VEC3F_ONES)
    : value(gain), texture(texture) {}

    inline Vec3f get_value(const float& u, const float& v, const float& w, Resources& resources) const
    {
        if(texture)
            return value * texture->get_vec3f(u, v, w, resources);
        return value;
    }

private:
    const Vec3f value;
    Texture * texture;
};

class AttributeFloat
{
public:
    AttributeFloat(const float &value)
    : value(value), texture(nullptr) {}
    AttributeFloat(Texture * texture, const float &gain = 1.f)
    : value(gain), texture(texture) {}

    inline float get_value(const float& u, const float& v, const float& w, Resources& resources) const
    {
        if(texture)
            return value * texture->get_float(u, v, w, resources);
        return value;
    }

private:
    const float value;
    Texture * texture;
};
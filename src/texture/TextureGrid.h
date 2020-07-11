#pragma once

#include <texture/Texture.h>

class TextureGrid : public Texture
{
public:
    TextureGrid(const Vec3f& fill, const Vec3f& line, const float& line_size, const Vec3f scale = VEC3F_ONES)
    : fill(fill), line(line), line_size(line_size), scale(scale) {}

    Vec3f evaluate(const float& u, const float& v, const float& w, const Intersect * intersect, Resources& resources) const
    {
        Vec3f uvw = scale * Vec3f(u, v, w);
        uvw = uvw - Vec3f::floor(uvw);
        
        float d = (VEC3F_ONES - uvw).min();

        return d < line_size ? line : fill;
    }

    virtual Vec3f delta() const { return 1.f / (line_size * 0.5f); };
    virtual bool null() const { return fill.null() && line.null(); };

private:
    const Vec3f fill;
    const Vec3f line;
    const float line_size;
    const Vec3f scale;
};

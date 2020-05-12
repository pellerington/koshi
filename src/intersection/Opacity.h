#pragma once

#include <base/Object.h>
#include <intersection/Intersect.h>
#include <Util/Attribute.h>

class Opacity : public Object
{
public:
    Opacity(const AttributeVec3f& opacity_attr, const bool& hide_camera) : opacity_attr(opacity_attr), hide_camera(hide_camera) {}

    inline Vec3f get_opacity(const Intersect * intersect, Resources &resources)
    {
        return (hide_camera && !intersect->path->depth) ? VEC3F_ZERO : opacity_attr.get_value(intersect->surface.u, intersect->surface.v, 0.f, resources);
    }
    
private:
    const AttributeVec3f opacity_attr;
    const bool hide_camera;
};

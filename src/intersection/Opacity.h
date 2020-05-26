#pragma once

#include <base/Object.h>
#include <intersection/Intersect.h>
#include <Util/Attribute.h>

// TODO: Need to clean this up and figure out how exacty it should work.
class Opacity : public Object
{
public:
    Opacity(const AttributeVec3f& opacity_attr, const bool& hide_camera) : opacity_attr(opacity_attr), hide_camera(hide_camera) {}

    // TODO: Need to seperate local and global visiblity so that, we can ignore volumes who are hide camera, but still evaluate opacity at specific points.

    inline Vec3f get_opacity(const float& u, const float& v, const float& w, const Intersect * intersect, Resources &resources)
    {
        return (hide_camera && !intersect->path->depth) ? VEC3F_ZERO : opacity_attr.get_value(u, v, w, resources);
    }
    
private:
    const AttributeVec3f opacity_attr;
    const bool hide_camera;
};

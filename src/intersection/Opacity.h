#pragma once

#include <base/Object.h>
#include <intersection/Intersect.h>

// TODO: Have the texture part, move this onto the material itself. Global parts move onto the geometry.
class Opacity : public Object
{
public:
    Opacity(const Texture * opacity, const bool& hide_camera) : opacity(opacity), hide_camera(hide_camera) {}

    // TODO: Need to seperate local and global visiblity so that, we can ignore volumes who are hide camera, but still evaluate opacity at specific points.

    inline Vec3f get_opacity(const float& u, const float& v, const float& w, const Intersect * intersect, Resources& resources)
    {
        return (hide_camera && !intersect->path->depth) ? VEC3F_ZERO : opacity->evaluate<Vec3f>(u, v, w, intersect, resources);
    }
    
private:
    const Texture * opacity;
    const bool hide_camera;
};

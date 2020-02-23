#pragma once

#include "../Util/Surface.h"
#include "../Util/Resources.h"

#include <iostream>

class Texture
{
public:
    virtual const Vec3f get_vec3f(const float &u, const float &v, const float &w, Resources &resources) { return false; }
    virtual const float get_float(const float &u, const float &v, const float &w, Resources &resources) { return false; }

    // virtual const Vec4f get_vec4f(const float &u, const float &v, const float &w, Resources &resources) { return false; }
    // virtual const float get_float(const Intersect &intersect, Resources &resources) { return get_float(intersect.surface.u, intersect.surface.v, 0.f, resources ); }

};

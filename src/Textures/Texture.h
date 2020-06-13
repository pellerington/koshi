#pragma once

#include <Util/Resources.h>
#include <base/Object.h>

#include <iostream>

class Texture : public Object
{
public:
    virtual Vec3f get_vec3f(const float &u, const float &v, const float &w, Resources &resources) { return false; }
    virtual float get_float(const float &u, const float &v, const float &w, Resources &resources) { return false; }

    // virtual const Vec3f get_vec3f(const Vec3f& uvw, const Intersect * intersect Resources &resources) { return false; }

    // TODO: Have spacial and procedural textures ect handle this.
    virtual Vec3f delta() const { return VEC3F_ZERO; }
};

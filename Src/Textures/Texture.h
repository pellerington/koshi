#pragma once

#include "../Util/Surface.h"

#include <iostream>

class Texture
{
public:
    virtual const bool get_vec3f(const Surface &surface, Vec3f &out) { return false; }
    virtual const bool get_vec3f(const float u, const float v, Vec3f &out) { return false; }

    // const bool get_vec4f(const Surface &surface, Vec4f &out) { return false; }
    virtual const bool get_float(const Surface &surface, float &out) { return false; }
    virtual const bool get_int(const Surface &surface, int &out) { return false; }

    // const bool get_rgb(const VolumePoint &volume_point?, Vec3f &out) { return false; }
};

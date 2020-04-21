#pragma once

#include <cfloat>
#include <unordered_set>
#include <Math/Types.h>

struct Ray
{
    Ray(const Vec3f &pos, const Vec3f &dir, const float& tmin = 0.f, const float& tmax = FLT_MAX)
    : pos(pos), dir(dir), inv_dir(1.f / dir), tmin(tmin), tmax(tmax) {}

    // Gets the position at t along the ray.
    inline Vec3f get_position(const float &t) const { return pos + t * dir; }
    inline Vec3f operator[](const float &t) const { return pos + t * dir; }

    // Basic info about our ray.
    const Vec3f pos;
    const Vec3f dir;
    const Vec3f inv_dir;

    // Optional ray info.
    const float tmin;
    const float tmax;
};
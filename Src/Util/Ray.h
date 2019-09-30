#pragma once

#include <cfloat>
#include "../Math/Types.h"

struct Ray
{
    Ray(const Vec3f &pos, const Vec3f &dir)
    : pos(pos), dir(dir), inv_dir(1.f / dir) {}
    inline Vec3f get_position(const float &_t) const { return pos + _t * dir; }
    const Vec3f pos;
    const Vec3f dir;
    const Vec3f inv_dir;
    // TODO: tmin/tmax
    float t = FLT_MAX;
    bool hit = false;
    uint depth = 0;
};

/*
inline bool get_refation(const Surface &surface, const double &eta, Vec3f &out)
{
    //Dont use surface pass in normal wi and ior_in
    float n_dot_wi = clamp(surface.normal.dot(surface.wi), -1.f, 1.f);
    float k = 1.f - eta * eta * (1.f - n_dot_wi * n_dot_wi);
    if(k < 0) return false;

    return eta * surface.wi + (eta * fabs(n_dot_wi) - sqrtf(k)) * ((surface.enter) ? surface.normal : -surface.normal);
}
*/

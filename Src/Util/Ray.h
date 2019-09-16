#pragma once

#include <cfloat>
#include "../Math/Types.h"

struct Ray
{
    Vec3f o;
    Vec3f dir;
    float t = FLT_MAX;
    bool hit = false;
    uint depth = 0;
    Vec3f inv_dir;
};

/*

Maybe put this is another file

inline bool get_refation(const Surface &surface, const double &eta, Vec3f &out)
{
    //Dont use surface pass in normal wi and ior_in
    float n_dot_wi = clamp(surface.normal.dot(surface.wi), -1.f, 1.f);
    float k = 1.f - eta * eta * (1.f - n_dot_wi * n_dot_wi);
    if(k < 0) return false;

    return eta * surface.wi + (eta * fabs(n_dot_wi) - sqrtf(k)) * ((surface.enter) ? surface.normal : -surface.normal);
}

*/

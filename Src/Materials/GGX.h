#pragma once

#include "../Math/Types.h"

inline float D(const Vec3f &n, const Vec3f &h, const double &n_dot_h, const double &roughness_sqr)
{
    // GGX Distribution ( https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf )
    const float tan_h = n.cross(h).length() / n_dot_h;
    const float det = roughness_sqr + (tan_h * tan_h);
    return (((n_dot_h > 0) ? 1.f : 0.f) * roughness_sqr) / (PI * std::pow(n_dot_h, 4) * det * det);
}

inline float G1(const Vec3f &v, const Vec3f &n, const Vec3f &h, const float &h_dot_v, const float &n_dot_v, const double &roughness_sqr)
{
    // GGX Geometric Term MaterialGGXReflect ( https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf )
    const float x = ((h_dot_v / n_dot_v) > 0.f) ? 1.f : 0.f;
    const float tan_sqr = std::pow(v.cross(n).length() / n_dot_v, 2);
    return (x * 2.f) / (1.f + sqrtf(1.f + roughness_sqr * tan_sqr));
}

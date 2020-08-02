#pragma once

#include <koshi/geometry/GeometryData.h>
class MaterialVolume;

struct Volume : public GeometryData
{
    Vec3f uvw_near, uvw_far;

    struct Segment {
        float t0, t1;
        Vec3f max_density, min_density;
        Segment * next;
    };
    Segment * segment = nullptr;
    // Add function to access sement using [t].

    // Vec3f opacity;

    MaterialVolume * material;
};
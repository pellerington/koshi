#pragma once

#include <geometry/GeometryData.h>
class MaterialVolume;

struct Volume : public GeometryData
{
    // TODO: Make this more generable.
    Vec3f uvw0, uvw1;

    struct Segment {
        float t0, t1;
        Vec3f max_density, min_density;
        Segment * next;
    };
    Segment * segment = nullptr;
    

    // Add function to access sement using [t].

    // Opacity???

    MaterialVolume * material;
};
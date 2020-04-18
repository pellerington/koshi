#pragma once

#include <memory>
#include <intersection/Ray.h>
#include <Util/Surface.h>
class Geometry;

// Make data type with pointers to next intersection in list????
struct Intersect
{
    Intersect(Ray& ray)
    : ray(ray), geometry(nullptr),
      surface(ray.dir, ray.ior->get_curr_ior(), ray.ior->get_prev_ior()),
      ior(ray.ior)
    {}

    Ray& ray;

    Geometry * geometry;

    Surface surface;
    const IorStack * ior;
};

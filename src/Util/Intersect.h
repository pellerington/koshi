#pragma once

#include <Util/Ray.h>

struct Intersect
{
    Intersect(Ray &ray)
    : ray(ray), object(nullptr),
      surface(ray.dir, ray.ior->get_curr_ior(), ray.ior->get_prev_ior()),
      volumes(ray, ray.in_volumes),
      ior(ray.ior)
    {}

    Ray &ray;

    // Replace this with material_ptr, emission_ptr and object_id???
    std::shared_ptr<Object> object;

    Surface surface;
    VolumeStack volumes;
    const IorStack * ior;
};

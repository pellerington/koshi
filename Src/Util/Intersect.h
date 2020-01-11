#pragma once

#include "Ray.h"

struct Intersect
{
    Intersect(Ray &ray)
    : ray(ray), object(nullptr), surface(ray.dir), volumes(ray, ray.in_volumes)
    {}

    Ray &ray;

    // Replace this with material_ptr, emission_ptr and object_id???
    std::shared_ptr<Object> object;

    Surface surface;
    VolumeStack volumes;
};

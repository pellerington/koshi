#pragma once

#include <intersection/Ray.h>

inline bool intersect_bbox(const Ray &ray, const Box3f &box)
{
    const Vec3f a = (box.min() - ray.pos) * ray.inv_dir;
    const Vec3f b = (box.max() - ray.pos) * ray.inv_dir;
    const float tmin = Vec3f::min(a, b).max();
    const float tmax = Vec3f::max(a, b).min();

    if (tmax < 0 || tmin > tmax || tmax < ray.tmin || tmin > ray.tmax)
        return false;

    return true;
}

inline bool intersect_bbox(const Ray &ray, const Box3f &box, float &t0, float &t1)
{
    const Vec3f a = (box.min() - ray.pos) * ray.inv_dir;
    const Vec3f b = (box.max() - ray.pos) * ray.inv_dir;
    const float tmin = Vec3f::min(a, b).max();
    const float tmax = Vec3f::max(a, b).min();

    if (tmax < 0 || tmin > tmax || tmax < ray.tmin || tmin > ray.tmax)
        return false;

    t0 = std::max(tmin, ray.tmin);
    t1 = std::min(tmax, ray.tmax);

    return true;
}
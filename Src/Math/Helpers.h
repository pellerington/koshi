#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "../Util/Ray.h"

inline const float clamp(const float &value, const float &min, const float &max)
{
    return (value < min) ? min : ((value > max) ? max : value);
}

inline const Eigen::Matrix3f world_transform(const Vec3f &n)
{
    Vec3f nu = Vec3f::Zero();
    if (std::fabs(n[0]) > std::fabs(n[1]))
        nu = Vec3f(n[2], 0, -n[0]) / sqrtf(n[0] * n[0] + n[2] * n[2]);
    else
        nu = Vec3f(0, -n[2], n[1]) / sqrtf(n[1] * n[1] + n[2] * n[2]);
    Vec3f nv = nu.cross(n);
    Eigen::Matrix3f transform;
    transform.col(0) = nv;
    transform.col(1) = n;
    transform.col(2) = nu;
    return transform;
}

// Move these into a custom BBOX class eventually
inline const float surface_area(const Eigen::AlignedBox3f &bbox)
{
    float surface_area = 0.f;
    surface_area += 2.f * bbox.sizes()[0] * bbox.sizes()[1];
    surface_area += 2.f * bbox.sizes()[0] * bbox.sizes()[2];
    surface_area += 2.f * bbox.sizes()[1] * bbox.sizes()[2];
    return surface_area;
}

// Move these into a custom BBOX class eventually
inline bool intersect_bbox(Ray &ray, const Eigen::AlignedBox3f &bbox)
{
    Vec3f t1 = (bbox.min() - ray.o).cwiseProduct(ray.inv_dir);
    Vec3f t2 = (bbox.max() - ray.o).cwiseProduct(ray.inv_dir);
    float tmin = (t1.cwiseMin(t2)).maxCoeff();
    float tmax = (t1.cwiseMax(t2)).minCoeff();

    if (tmax < 0 || tmin > tmax || tmin > ray.t)
        return false;

    return true;
}

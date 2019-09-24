#pragma once

#include "Vec3f.h"
#include "../Util/Ray.h"

class Box3f
{
public:

    Box3f() : box_min(Vec3f(FLT_MAX)), box_max(Vec3f(FLT_MIN)), box_center(0.f), box_length(Vec3f(0.f)) {}
    Box3f(const Vec3f &_min, const Vec3f &_max) : box_min(_min), box_max(_max), box_center((_max + _min) / 2.f), box_length(_max - _min) {}

    inline const Vec3f& min() const { return box_min; }
    inline const Vec3f& max() const { return box_max; }
    inline const Vec3f& center() const { return box_center; }
    inline const Vec3f& length() const { return box_length; }
    inline const float surface_area() const
    {
        float surface_area = 0.f;
        surface_area += 2.f * box_length[0] * box_length[1];
        surface_area += 2.f * box_length[0] * box_length[2];
        surface_area += 2.f * box_length[1] * box_length[2];
        return surface_area;
    }

    void extend(const Box3f &box)
    {
        box_min.min(box.min());
        box_max.max(box.max());
        box_center = (box_max + box_min) / 2.f;
        box_length = box_max - box_min;
    }

    inline bool intersect(const Ray &ray)
    {
        const Vec3f t1 = (box_min - ray.pos) * ray.inv_dir;
        const Vec3f t2 = (box_max - ray.pos) * ray.inv_dir;
        const float tmin = Vec3f::min(t1, t2).max();
        const float tmax = Vec3f::max(t1, t2).min();

        // Return tmin/tmax?
        if (tmax < 0 || tmin > tmax || tmin > ray.t)
            return false;

        return true;
    }

private:

    Vec3f box_min;
    Vec3f box_max;
    Vec3f box_center;
    Vec3f box_length;
    // float box_volume;
};

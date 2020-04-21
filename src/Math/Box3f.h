#pragma once

#include <cfloat>
#include <Math/Vec3f.h>

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
        box_center = (box_max + box_min) * 0.5f;
        box_length = box_max - box_min;
    }

    void extend(const Vec3f &v)
    {
        box_min.min(v);
        box_max.max(v);
        box_center = (box_max + box_min) * 0.5f;
        box_length = box_max - box_min;
    }

    friend std::ostream& operator<<(std::ostream& os, const Box3f& b)
    {
        os << "min: (" << b.box_min[0] << " " << b.box_min[1] << " " << b.box_min[2] << ")\n";
        os << "max: (" << b.box_max[0] << " " << b.box_max[1] << " " << b.box_max[2] << ")";
        return os;
    }

private:

    Vec3f box_min;
    Vec3f box_max;
    Vec3f box_center;
    Vec3f box_length;
};

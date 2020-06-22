#pragma once

#include <geometry/Geometry.h>

#define AREA_LENGTH 1.f

class GeometryArea : public Geometry
{
public:
    GeometryArea(const Transform3f &obj_to_world);    

    inline const Vec3f& get_world_normal() { return world_normal; }

private:
    Vec3f world_normal;

    static const Box3f bbox;
};


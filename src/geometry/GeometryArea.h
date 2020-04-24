#pragma once

#include <geometry/Geometry.h>

class GeometryArea : public Geometry
{
public:
    GeometryArea(const Transform3f &obj_to_world);    

    static const Vec3f vertices[8];
    inline const Vec3f& get_world_normal() { return world_normal; }

private:
    Vec3f world_normal;
};


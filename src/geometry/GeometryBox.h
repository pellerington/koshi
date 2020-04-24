#pragma once

#include <geometry/Geometry.h>

class GeometryBox : public Geometry
{
public:
    GeometryBox(const Transform3f &obj_to_world);
    
    static const Vec3f vertices[8];
    static const uint indices[6][4];
};

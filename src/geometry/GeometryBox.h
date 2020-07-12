#pragma once

#include <geometry/Geometry.h>

#define BOX_LENGTH 1.f

class GeometryBox : public Geometry
{
public:
    GeometryBox(const Transform3f& obj_to_world, const GeometryVisibility& visibility);
    
private:
    static const Box3f bbox;
};

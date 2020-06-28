#pragma once

#include <geometry/Geometry.h>

#define AREA_LENGTH 1.f

class GeometryArea : public Geometry
{
public:
    GeometryArea(const Transform3f& obj_to_world);
private:
    static const Box3f bbox;
};


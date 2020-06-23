#pragma once

#include <geometry/Geometry.h>

#define SPHERE_RADIUS 1.f
#define SPHERE_RADIUS_SQR SPHERE_RADIUS*SPHERE_RADIUS

class GeometrySphere : public Geometry
{
public:
    GeometrySphere(const Transform3f &obj_to_world);

protected:
    static const Box3f bbox; 
};

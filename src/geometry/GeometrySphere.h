#pragma once

#include <geometry/Geometry.h>

#define SPHERE_RADIUS 1.f

class GeometrySphere : public Geometry
{
public:
    GeometrySphere(const Transform3f &obj_to_world);

    // TODO: Remove these when we combine sphere sampling techniques. and/or move them to the sampler itself. 
    inline const Vec3f& get_world_center() { return center; }
    inline const float& get_world_radius() { return radius; } 
    inline const float& get_world_radius_sqr() { return radius_sqr; } 
    inline const bool& is_elliptoid() { return elliptoid; }

protected:
    Vec3f center;
    float radius, radius_sqr;
    bool elliptoid;

    static const Box3f bbox; 
};

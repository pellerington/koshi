#pragma once

#include <geometry/Geometry.h>

class GeometrySphere : public Geometry
{
public:
    GeometrySphere(const Transform3f &obj_to_world = Transform3f(), std::shared_ptr<Material> material = nullptr,
                    std::shared_ptr<Light> light = nullptr, const bool hide_camera = false);

    // Remove some of these when we move to transform based intersections.
    inline const Vec3f& get_world_center() { return center; }
    inline const float& get_world_radius() { return radius; } 
    inline const float& get_world_radius_sqr() { return radius_sqr; } 
    inline const bool& is_elliptoid() { return elliptoid; }

protected:
    Vec3f center;
    float radius, radius_sqr;
    bool elliptoid;
};

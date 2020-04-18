#pragma once

#include <geometry/Geometry.h>

class GeometrySphere : public Geometry
{
public:
    GeometrySphere(const Transform3f &obj_to_world = Transform3f(), std::shared_ptr<Material> material = nullptr,
                    std::shared_ptr<Light> light = nullptr, const bool hide_camera = false);

    // static void intersect_callback(const RTCIntersectFunctionNArguments* args);

    // Remove some of these when we move to transform based intersections.
    inline const Vec3f& get_center() { return center; }
    inline const float& get_radius_sqr() { return radius_sqr; } 
    inline const bool& is_elliptoid() { return elliptoid; }

protected:
    Vec3f center;
    float x_len, y_len, z_len;
    float radius, radius_sqr;
    float area;
    bool elliptoid;
};

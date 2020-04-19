#include <geometry/GeometrySphere.h>

GeometrySphere::GeometrySphere(const Transform3f &obj_to_world, std::shared_ptr<Material> material, std::shared_ptr<Light> light, const bool hide_camera)
: Geometry(obj_to_world, light, material, hide_camera)
{
    bbox = obj_to_world * Box3f(Vec3f(-1.f), Vec3f(1.f));
    center = obj_to_world * Vec3f(0.f);

    const float x_len = obj_to_world.multiply(Vec3f(1.f, 0.f, 0.f), false).length();
    const float y_len = obj_to_world.multiply(Vec3f(0.f, 1.f, 0.f), false).length();
    const float z_len = obj_to_world.multiply(Vec3f(0.f, 0.f, 1.f), false).length();
    radius = std::max(x_len, std::max(y_len, z_len));
    radius_sqr = radius * radius;
    
    elliptoid = fabs(x_len - y_len) > 0.01f || fabs(x_len - z_len) > 0.01f || fabs(z_len - y_len) > 0.01f;
}


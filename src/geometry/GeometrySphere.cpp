#include <geometry/GeometrySphere.h>

const Box3f GeometrySphere::bbox = Box3f(Vec3f(-SPHERE_RADIUS), Vec3f(SPHERE_RADIUS)); 

GeometrySphere::GeometrySphere(const Transform3f &obj_to_world)
: Geometry(obj_to_world)
{
    obj_bbox = bbox;
    world_bbox = obj_to_world * obj_bbox;

    // TODO: Rename these as world_xxx.
    center = obj_to_world * Vec3f(0.f);
    const float x_len = obj_to_world.multiply(Vec3f(SPHERE_RADIUS, 0.f, 0.f), false).length();
    const float y_len = obj_to_world.multiply(Vec3f(0.f, SPHERE_RADIUS, 0.f), false).length();
    const float z_len = obj_to_world.multiply(Vec3f(0.f, 0.f, SPHERE_RADIUS), false).length();
    radius = std::max(x_len, std::max(y_len, z_len));
    radius_sqr = radius * radius;
    
    elliptoid = fabs(x_len - y_len) > 0.01f || fabs(x_len - z_len) > 0.01f || fabs(z_len - y_len) > 0.01f;
}


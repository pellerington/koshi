#include <koshi/geometry/GeometrySphere.h>

const Box3f GeometrySphere::bbox = Box3f(Vec3f(-SPHERE_RADIUS), Vec3f(SPHERE_RADIUS)); 

GeometrySphere::GeometrySphere(const Transform3f& obj_to_world, const GeometryVisibility& visibility)
: Geometry(obj_to_world, visibility)
{
    obj_bbox = bbox;
    world_bbox = obj_to_world * obj_bbox;
}


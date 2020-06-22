#include <geometry/GeometryArea.h>

const Box3f GeometryArea::bbox = Box3f(Vec3f(-AREA_LENGTH*0.5f, -AREA_LENGTH*0.5f, 0.f), Vec3f(AREA_LENGTH*0.5f, AREA_LENGTH*0.5f, 0.f));

GeometryArea::GeometryArea(const Transform3f &obj_to_world)
: Geometry(obj_to_world)
{
    obj_bbox = bbox;
    world_bbox = obj_to_world * obj_bbox;
    world_normal = obj_to_world.multiply(Vec3f(0.f, 0.f, -1.f), false);
    world_normal.normalize();
}

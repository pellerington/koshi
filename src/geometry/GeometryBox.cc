#include <geometry/GeometryBox.h>

const Box3f GeometryBox::bbox = Box3f(Vec3f(-BOX_LENGTH*0.5f), Vec3f(BOX_LENGTH*0.5f));

GeometryBox::GeometryBox(const Transform3f &obj_to_world)
: Geometry(obj_to_world)
{
    obj_bbox = bbox;
    world_bbox = obj_to_world * bbox;
}

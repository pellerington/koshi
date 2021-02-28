#include <koshi/geometry/GeometryQuad.h>

KOSHI_OPEN_NAMESPACE

GeometryQuad::GeometryQuad() : Geometry(Type::QUAD) 
{

}

void GeometryQuad::setTransform(const Transform& _obj_to_world)
{
    Geometry::setTransform(_obj_to_world);

}

KOSHI_CLOSE_NAMESPACE
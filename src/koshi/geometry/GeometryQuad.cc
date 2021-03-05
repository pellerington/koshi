#include <koshi/geometry/GeometryQuad.h>

#include <koshi/OptixHelpers.h>

KOSHI_OPEN_NAMESPACE

const float GeometryQuad::vertices[4*3] = { -0.5f, 0.5f, 0.f, 0.5f, 0.5f, 0.f, 0.5f, -0.5f, 0.f, -0.5f, -0.5f, 0.f };
const uint32_t GeometryQuad::indices[2*3] = { 0, 1, 2, 2, 3, 0 };

GeometryQuad::GeometryQuad() : Geometry(Type::QUAD) 
{
}

// void GeometryQuad::setTransform(const Transform& _obj_to_world)
// {
//     Geometry::setTransform(_obj_to_world);
//     Set normal???
// }

KOSHI_CLOSE_NAMESPACE
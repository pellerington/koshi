#include <geometry/GeometryArea.h>

const Vec3f GeometryArea::vertices[8] = 
{
    Vec3f( 0.5f,  0.5f, 0.f),
    Vec3f( 0.5f, -0.5f, 0.f),
    Vec3f(-0.5f, -0.5f, 0.f),
    Vec3f(-0.5f,  0.5f, 0.f)
};

GeometryArea::GeometryArea(const Transform3f &obj_to_world, std::shared_ptr<Material> material, std::shared_ptr<Light> light)
: Geometry(obj_to_world, light, material)
{
    bbox = obj_to_world * Box3f(vertices[2], vertices[0]);
    world_normal = obj_to_world.multiply(Vec3f(0.f, 0.f, -1.f), false);
    world_normal.normalize();
}

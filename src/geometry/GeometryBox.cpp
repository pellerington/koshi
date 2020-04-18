#include <geometry/GeometryBox.h>

const Vec3f GeometryBox::vertices[8] = 
{
    Vec3f(-0.5f, -0.5f, -0.5f),
    Vec3f( 0.5f, -0.5f, -0.5f),
    Vec3f( 0.5f, -0.5f,  0.5f),
    Vec3f(-0.5f, -0.5f,  0.5f),
    Vec3f(-0.5f,  0.5f, -0.5f),
    Vec3f( 0.5f,  0.5f, -0.5f),
    Vec3f( 0.5f,  0.5f,  0.5f),
    Vec3f(-0.5f,  0.5f,  0.5f)
};

const uint GeometryBox::indices[6][4] = 
{
    {0, 4, 5, 1},
    {1, 5, 6, 2},
    {2, 6, 7, 3},
    {0, 3, 7, 4},
    {4, 7, 6, 5},
    {0, 1, 2, 3},
};

GeometryBox::GeometryBox(const Transform3f &obj_to_world, std::shared_ptr<Material> material)
: Geometry(obj_to_world, nullptr, material)
{
    bbox = obj_to_world * Box3f(Vec3f(-0.5), Vec3f(0.5));
}

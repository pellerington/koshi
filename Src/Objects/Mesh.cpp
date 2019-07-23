#include "Mesh.h"

#include <iostream>

Mesh::Mesh(std::vector<std::shared_ptr<Vec3f>> &vert, std::vector<std::shared_ptr<Triangle>> &triangle, std::shared_ptr<Material> material)
: Object(material), vert(std::move(vert)), triangle(std::move(triangle))
{
}

Mesh::Mesh(std::vector<std::shared_ptr<Vec3f>> &vert, std::vector<std::shared_ptr<Vec3f>> &normals, std::vector<std::shared_ptr<Triangle>> &triangle, std::shared_ptr<Material> material)
: Object(material), vert(std::move(vert)), normals(normals), triangle(std::move(triangle))
{
}

void Mesh::add_material(std::shared_ptr<Material> _material)
{
    for(size_t i = 0; i < triangle.size(); i++)
        triangle[i]->add_material(_material);
}


bool Mesh::intersect(Ray &ray, Surface &surface)
{
    for(size_t i = 0; i < triangle.size(); i++)
    {
        triangle[i]->intersect(ray, surface);
    }
    return ray.hit;
}

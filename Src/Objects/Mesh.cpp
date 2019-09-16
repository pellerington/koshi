#include "Mesh.h"

#if !EMBREE

#include <iostream>

Mesh::Mesh(std::vector<Vec3f> &_vertices, std::vector<TriangleData> &triangle_data, std::shared_ptr<Material> material)
: Object(material), vertices(std::move(_vertices))
{
    for(uint i = 0; i < triangle_data.size(); i++)
    {
        triangles.emplace_back(new Triangle(
            std::shared_ptr<Vec3f>(&vertices[triangle_data[i].v_index[0]]),
            std::shared_ptr<Vec3f>(&vertices[triangle_data[i].v_index[1]]),
            std::shared_ptr<Vec3f>(&vertices[triangle_data[i].v_index[2]]),
            nullptr, nullptr, nullptr, material
        ));
    }
    // Need to set bbox
}

Mesh::Mesh(std::vector<Vec3f> &_vertices, std::vector<Vec3f> &_normals, std::vector<TriangleData> &triangle_data, std::shared_ptr<Material> material)
: Object(material), vertices(std::move(_vertices)), normals(std::move(_normals))
{
    for(uint i = 0; i < triangle_data.size(); i++)
    {
        triangles.emplace_back(new Triangle(
            std::shared_ptr<Vec3f>(&vertices[triangle_data[i].v_index[0]]),
            std::shared_ptr<Vec3f>(&vertices[triangle_data[i].v_index[1]]),
            std::shared_ptr<Vec3f>(&vertices[triangle_data[i].v_index[2]]),
            std::shared_ptr<Vec3f>(&normals[triangle_data[i].n_index[0]]),
            std::shared_ptr<Vec3f>(&normals[triangle_data[i].n_index[1]]),
            std::shared_ptr<Vec3f>(&normals[triangle_data[i].n_index[2]]),
            material
        ));
    }
    // Need to set bbox
}

bool Mesh::intersect(Ray &ray, Surface &surface)
{
    for(size_t i = 0; i < triangles.size(); i++)
    {
        triangles[i]->intersect(ray, surface);
    }
    return ray.hit;
}

std::vector<std::shared_ptr<Object>> Mesh::get_sub_objects()
{
    return std::vector<std::shared_ptr<Object>>(triangles.begin(), triangles.end());
}

#endif

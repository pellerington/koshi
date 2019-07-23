#pragma once

#include "Triangle.h"

#include <vector>

class Triangle;

class Mesh : public Object
{
public:
    Mesh(std::vector<std::shared_ptr<Vec3f>> &vert, std::vector<std::shared_ptr<Triangle>> &triangle, std::shared_ptr<Material> material = nullptr);
    Mesh(std::vector<std::shared_ptr<Vec3f>> &vert, std::vector<std::shared_ptr<Vec3f>> &normals, std::vector<std::shared_ptr<Triangle>> &triangle, std::shared_ptr<Material> material = nullptr);
    ObjectType get_type() { return ObjectType::Mesh; }

    std::vector<std::shared_ptr<Object>> get_sub_objects() { return std::vector<std::shared_ptr<Object>>(triangle.begin(), triangle.end()); }

    void add_material(std::shared_ptr<Material> _material);

    bool intersect(Ray &ray, Surface &surface);
private:
    std::vector<std::shared_ptr<Vec3f>> vert;
    std::vector<std::shared_ptr<Vec3f>> normals;
    std::vector<std::shared_ptr<Triangle>> triangle;
};

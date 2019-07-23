#pragma once

#include "Object.h"

#include <memory>

class Triangle : public Object
{
public:
    Triangle(std::shared_ptr<Vec3f> v0, std::shared_ptr<Vec3f> v1, std::shared_ptr<Vec3f> v2,
             std::shared_ptr<Vec3f> n0 = nullptr, std::shared_ptr<Vec3f> n1 = nullptr, std::shared_ptr<Vec3f> n2 = nullptr,
             std::shared_ptr<Material> material = nullptr);
    Triangle(Vec3f v0, Vec3f v1, Vec3f v2, std::shared_ptr<Material> material = nullptr);
    Triangle(Vec3f v0, Vec3f v1, Vec3f v2, Vec3f n0, Vec3f n1, Vec3f n2, std::shared_ptr<Material> material = nullptr);
    ObjectType get_type() { return ObjectType::Triangle; }
    std::vector<std::shared_ptr<Object>> get_sub_objects() { return std::vector<std::shared_ptr<Object>>(1, std::shared_ptr<Object>(this)); }
    bool intersect(Ray &ray, Surface &surface);
    void init();
private:
    std::shared_ptr<Vec3f> vert[3];
    std::shared_ptr<Vec3f> normals[3];
    Vec3f normal;
    bool smooth_normals;
};

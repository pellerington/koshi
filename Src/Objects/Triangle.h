#pragma once

#include "Object.h"

#include <memory>

class Triangle : public Object
{
public:
    Triangle(const Vec3f &v0, const Vec3f &v1, const Vec3f &v2, std::shared_ptr<Material> material = nullptr);
    Triangle(const Vec3f &v0, const Vec3f &v1, const Vec3f &v2, const Vec3f &n0, const Vec3f &n1, const Vec3f &n2, std::shared_ptr<Material> material = nullptr);
    ObjectType get_type() { return ObjectType::Triangle; }
    Surface process_intersection(const RTCRayHit &rtcRayHit, const Ray &ray);

private:
    void init();
    std::shared_ptr<Vec3f> vertices[3];
    std::shared_ptr<Vec3f> normals[3];
    Vec3f normal;
    bool smooth_normals;
};

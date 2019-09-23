#pragma once

#include "Object.h"

#include <memory>

class Triangle : public Object
{
public:
    Triangle(std::shared_ptr<Vec3f> v0, std::shared_ptr<Vec3f> v1, std::shared_ptr<Vec3f> v2,
             std::shared_ptr<Vec3f> n0 = nullptr, std::shared_ptr<Vec3f> n1 = nullptr, std::shared_ptr<Vec3f> n2 = nullptr,
             std::shared_ptr<Material> material = nullptr);
    Triangle(const Vec3f &v0, const Vec3f &v1, const Vec3f &v2, std::shared_ptr<Material> material = nullptr);
    Triangle(const Vec3f &v0, const Vec3f &v1, const Vec3f &v2, const Vec3f &n0, const Vec3f &n1, const Vec3f &n2, std::shared_ptr<Material> material = nullptr);
    ObjectType get_type() { return ObjectType::Triangle; }
    std::vector<std::shared_ptr<Object>> get_sub_objects() { return std::vector<std::shared_ptr<Object>>(1, std::shared_ptr<Object>(this)); }
    bool intersect(Ray &ray, Surface &surface);

#if EMBREE
    void process_intersection(const RTCRayHit &rtcRayHit, Ray &ray, Surface &surface);
#endif

private:
    void init();
    std::shared_ptr<Vec3f> vertices[3];
    std::shared_ptr<Vec3f> normals[3];
    Vec3f normal;
    bool smooth_normals;
};

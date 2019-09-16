#pragma once

#include "Object.h"

class Sphere : public Object
{
public:
    Sphere(Vec3f position, float scale, std::shared_ptr<Material> material = nullptr);
    ObjectType get_type() { return ObjectType::Sphere; }
    std::vector<std::shared_ptr<Object>> get_sub_objects() { return std::vector<std::shared_ptr<Object>>(1, std::shared_ptr<Object>(this)); }
    bool intersect(Ray &ray, Surface &surface);

#if EMBREE
    void process_intersection(RTCRayHit &rtcRayHit, Ray &ray, Surface &surface);
#endif

private:
    Vec3f position;
    float scale;
    float scale_sqr;
};

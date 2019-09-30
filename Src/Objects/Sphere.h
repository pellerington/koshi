#pragma once

#include "Object.h"

class Sphere : public Object
{
public:
    Sphere(const Vec3f &position, const float &scale, std::shared_ptr<Material> material = nullptr);
    ObjectType get_type() { return ObjectType::Sphere; }
    Surface process_intersection(const RTCRayHit &rtcRayHit, const Ray &ray);

private:
    Vec3f position;
    float scale;
    float scale_sqr;
};

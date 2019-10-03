#pragma once

#include "Object.h"

class ObjectSphere : public Object
{
public:
    ObjectSphere(const Vec3f &position, const float &scale, std::shared_ptr<Material> material = nullptr);
    Type get_type() { return Object::Sphere; }
    Surface process_intersection(const RTCRayHit &rtcRayHit, const Ray &ray);

private:
    Vec3f position;
    float scale;
    float scale_sqr;
};

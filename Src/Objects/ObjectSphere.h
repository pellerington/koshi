#pragma once

#include "Object.h"

class ObjectSphere : public Object
{
public:
    ObjectSphere(std::shared_ptr<Material> material = nullptr, const Transform3f &obj_to_world = Transform3f(), std::shared_ptr<Volume> volume = nullptr);
    Type get_type() { return Object::Sphere; }
    Surface process_intersection(const RTCRayHit &rtcRayHit, const Ray &ray);

private:
    Vec3f position;
    float scale;
};

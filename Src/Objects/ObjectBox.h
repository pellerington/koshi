#pragma once

#include "Object.h"

class ObjectBox : public Object
{
public:
    ObjectBox(std::shared_ptr<Material> material = nullptr, const Transform3f &obj_to_world = Transform3f(), std::shared_ptr<VolumeProperties> volume = nullptr);

    Type get_type() { return Object::Box; }
    Surface process_intersection(const RTCRayHit &rtcRayHit, const Ray &ray);
};

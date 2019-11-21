#pragma once

#include "Object.h"

class ObjectSphere : public Object
{
public:
    ObjectSphere(const Transform3f &obj_to_world = Transform3f(), std::shared_ptr<Material> material = nullptr, std::shared_ptr<Volume> volume = nullptr);
    Type get_type() { return Object::Sphere; }

private:
    Vec3f center;
};

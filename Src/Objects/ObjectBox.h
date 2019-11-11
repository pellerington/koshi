#pragma once

#include "Object.h"

class ObjectBox : public Object
{
public:
    ObjectBox(const Transform3f &obj_to_world = Transform3f(), std::shared_ptr<Material> material = nullptr, std::shared_ptr<Volume> volume = nullptr);
    Type get_type() { return Object::Box; }
};

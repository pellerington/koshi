#pragma once

#include <koshi/Transform.h>

KOSHI_OPEN_NAMESPACE

class Geometry
{
public:
    void setTransform(const Transform& _obj_to_world)
    {
        obj_to_world = _obj_to_world;
        world_to_obj = obj_to_world.inverse();
    }
    virtual ~Geometry() = default;
protected:
    Transform obj_to_world;
    Transform world_to_obj;
};

KOSHI_CLOSE_NAMESPACE
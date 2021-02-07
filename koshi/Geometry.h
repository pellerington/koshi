#pragma once

#include <koshi/math/Transform.h>

KOSHI_OPEN_NAMESPACE

enum GeometryType
{
    MESH
};

class Geometry
{
public:
    virtual GeometryType type() = 0;
    void setTransform(const Transform& _obj_to_world)
    {
        obj_to_world = _obj_to_world;
        world_to_obj = obj_to_world.inverse();
    }
    DEVICE_FUNCTION const Transform& get_obj_to_world() { return obj_to_world; }
    DEVICE_FUNCTION const Transform& get_world_to_obj() { return world_to_obj; }
protected:
    Transform obj_to_world;
    Transform world_to_obj;
};

KOSHI_CLOSE_NAMESPACE
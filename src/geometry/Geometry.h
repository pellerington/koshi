#pragma once

#include <memory>
#include <iostream>
#include <vector>

#include <base/Object.h>

#include <Util/Resources.h>
#include <intersection/Ray.h>
#include <lights/Light.h>

class Material;

class Geometry : public Object
{
public:
    Geometry(const Transform3f &obj_to_world = Transform3f())
    : obj_to_world(obj_to_world), 
      world_to_obj(Transform3f::inverse(obj_to_world)) {}

    inline const Box3f& get_bbox() { return bbox; }
    inline const Transform3f& get_obj_to_world() { return obj_to_world; }
    inline const Transform3f& get_world_to_obj() { return world_to_obj; }

protected:
    Box3f bbox;
    Transform3f obj_to_world;
    Transform3f world_to_obj;
};

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
    Geometry(const Transform3f &obj_to_world = Transform3f(),
             const bool &hide_camera = false)
    : obj_to_world(obj_to_world), 
      world_to_obj(Transform3f::inverse(obj_to_world)),
      hide_camera(hide_camera) 
    {
    }

    inline const Box3f& get_bbox() { return bbox; }
    inline const Transform3f& get_obj_to_world() { return obj_to_world; }
    inline const Transform3f& get_world_to_obj() { return world_to_obj; }

    // const Vec3f get_opacity(/* intersection */) { return hide_camera ? VEC3F_ZERO : VEC3F_ONES; }

protected:
    Box3f bbox;
    Transform3f obj_to_world;
    Transform3f world_to_obj;

    bool hide_camera;
};

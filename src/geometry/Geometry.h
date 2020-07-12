#pragma once

#include <memory>
#include <iostream>
#include <vector>

#include <base/Object.h>
#include <Util/Resources.h>
#include <math/Transform3f.h>

class GeometryAttribute 
{
public:
    virtual Vec3f evaluate(const float& u, const float& v, const float& w, const uint& prim, Resources& resources) const = 0;
    virtual ~GeometryAttribute() = default;
};

struct GeometryVisibility
{
    bool hide_camera = false;
};

class Geometry : public Object
{
public:
    Geometry(const Transform3f& obj_to_world, const GeometryVisibility& visibility)
    : obj_to_world(obj_to_world), 
      world_to_obj(Transform3f::inverse(obj_to_world)),
      visibility(visibility)
    {
    }

    inline const Box3f& get_obj_bbox() { return obj_bbox; }
    inline const Box3f& get_world_bbox() { return world_bbox; }

    inline const Transform3f& get_obj_to_world() { return obj_to_world; }
    inline const Transform3f& get_world_to_obj() { return world_to_obj; }

    const GeometryVisibility& get_visibility() { return visibility; }

    virtual const GeometryAttribute * get_geometry_attribute(const std::string& attribute_name) { return nullptr; }

protected:
    Box3f obj_bbox;
    Box3f world_bbox;
    Transform3f obj_to_world;
    Transform3f world_to_obj;
    GeometryVisibility visibility;
};

#pragma once

#include <geometry/Geometry.h>

class GeometryArea : public Geometry
{
public:
    GeometryArea(const Transform3f &obj_to_world = Transform3f(), std::shared_ptr<Material> material = nullptr, std::shared_ptr<Light> light = nullptr);    

    static const Vec3f vertices[8];
    inline const Vec3f& get_world_normal() { return world_normal; }

private:
    Vec3f world_normal;
};


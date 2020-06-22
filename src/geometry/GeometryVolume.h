#pragma once

#include <geometry/Geometry.h>

#define VOLUME_LENGTH 1.f

class GeometryVolume : public Geometry
{
public:
    GeometryVolume(const Transform3f &obj_to_world = Transform3f());

    void pre_render(Resources& resources);

    inline const Box3f& get_obj_bbox() { return obj_bbox; }

    struct VolumeBound {
        Box3f bbox;
        Vec3f max_density;
        Vec3f min_density;
    };
    inline const std::vector<VolumeBound>& get_bound() { return bounds; }

private:
    std::vector<VolumeBound> bounds;

    const static Box3f bbox;
};
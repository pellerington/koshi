#pragma once

#include <geometry/Geometry.h>

#define VOLUME_LENGTH 1.f

// class GeometryVolumeAttribute : public GeometryAttribute
// {
//     // virtual Vec3f resolution? delta?
// }

class GeometryVolume : public Geometry
{
public:
    GeometryVolume(const Transform3f& obj_to_world = Transform3f());

    void pre_render(Resources& resources);

    inline const Box3f& get_obj_bbox() { return obj_bbox; }

    struct VolumeBox3f {
        Box3f bbox;
        Vec3f max_density;
        Vec3f min_density;
    };
    inline const std::vector<VolumeBox3f>& get_bound() { return bounds; }

private:
    std::vector<VolumeBox3f> bounds;

    const static Box3f bbox;
};
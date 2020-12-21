#pragma once

#include <koshi/geometry/Geometry.h>

#define VOLUME_LENGTH 1.f

#define VOLUME_BVH_SPLITS 16


// class GeometryVolumeAttribute : public GeometryAttribute
// {
//     // virtual Vec3f resolution? delta?
// }

class GeometryVolume : public Geometry
{
public:
    GeometryVolume(const Transform3f& obj_to_world, const GeometryVisibility& visibility);

    void pre_render(Resources& resources);

    inline const Box3f& get_obj_bbox() { return obj_bbox; }

    struct VolumeBox3f {
        Box3f bbox;
        Vec3f max_density;
        Vec3f min_density;
    };
    inline const std::vector<VolumeBox3f>& get_acceleration_structure() { return acceleration_structure; }

private:
    void split_acceleration_structure(Vec3f max_density[VOLUME_BVH_SPLITS][VOLUME_BVH_SPLITS][VOLUME_BVH_SPLITS], Vec3f min_density[VOLUME_BVH_SPLITS][VOLUME_BVH_SPLITS][VOLUME_BVH_SPLITS], const Vec3u& n0, const Vec3u& n1);
    std::vector<VolumeBox3f> acceleration_structure;
    const static Box3f bbox;
};
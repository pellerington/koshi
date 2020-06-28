#include <geometry/GeometryVolume.h>

#include <material/MaterialVolume.h>

const Box3f GeometryVolume::bbox = Box3f(Vec3f(-VOLUME_LENGTH*0.5f), Vec3f(VOLUME_LENGTH*0.5f));

GeometryVolume::GeometryVolume(const Transform3f& obj_to_world)
: Geometry(obj_to_world)
{
    obj_bbox = bbox;
    world_bbox = obj_to_world * obj_bbox;
}

void GeometryVolume::pre_render(Resources& resources)
{
    MaterialVolume * material = get_attribute<MaterialVolume>("material");

    if(!material)
        return;

    if(material->homogenous())
    {
        VolumeBox3f b;
        b.bbox = obj_bbox;
        b.max_density = b.min_density = material->get_density(VEC3F_ZERO, nullptr, resources);
        bounds.push_back(b);
    }
    else
    {
        Vec3f delta = material->get_density_texture()->delta();
        if(!(delta > 0.f)) delta = 0.03125f; // Delta for procedurals.

        VolumeBox3f b;
        b.bbox = obj_bbox;
        b.max_density = FLT_MIN;
        b.min_density = FLT_MAX;

        Vec3f uvw;
        for(uvw.u = delta.u * 0.5f; uvw.u < 1.f; uvw.u += delta.u)
        for(uvw.v = delta.v * 0.5f; uvw.v < 1.f; uvw.v += delta.v)
        for(uvw.z = delta.w * 0.5f; uvw.w < 1.f; uvw.w += delta.w)
        {
            const Vec3f density = material->get_density(uvw, nullptr, resources);
            b.max_density.max(density);
            b.min_density.max(density);
        }

        bounds.push_back(b);
    }
}

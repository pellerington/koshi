#include <geometry/GeometryVolume.h>

#include <material/MaterialVolume.h>

GeometryVolume::GeometryVolume(const Transform3f &obj_to_world)
: Geometry(obj_to_world)
{
    obj_bbox = Box3f(Vec3f(-0.5), Vec3f(0.5));
    bbox = obj_to_world * obj_bbox;
}

void GeometryVolume::pre_render(Resources& resources)
{
    MaterialVolume * material = get_attribute<MaterialVolume>("material");

    if(!material)
        return;

    if(material->homogenous())
    {
        VolumeBound b;
        b.bbox = obj_bbox;
        b.max_density = b.min_density = material->get_density(VEC3F_ZERO, nullptr, nullptr, resources);
        bounds.push_back(b);
    }
    else
    {
        
    }
}

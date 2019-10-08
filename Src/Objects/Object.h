#pragma once

#include <memory>
#include <iostream>
#include <vector>

#include "../Scene/Embree.h"
#include "../Materials/Material.h"
#include "../Util/Surface.h"
#include "../Util/Ray.h"
#include "../Volume/VolumeStack.h"
class Surface;
class Material;

class Object
{
public:
    Object() : material(nullptr), obj_to_world(Transform3f()), world_to_obj(Transform3f()) {}
    Object(std::shared_ptr<Material> material, const Transform3f &obj_to_world, std::shared_ptr<VolumeProperties> volume = nullptr)
    : volume(volume), material(material), obj_to_world(obj_to_world), world_to_obj(Transform3f::inverse(obj_to_world)) {}

    std::shared_ptr<VolumeProperties> volume;

    enum Type
    {
        Triangle,
        Mesh,
        Sphere,
        Box
    };
    virtual Type get_type() = 0;

    const Box3f get_bbox() { return bbox; };

    virtual Surface process_intersection(const RTCRayHit &rtcRayHit, const Ray &ray) = 0;
    virtual void process_volume_intersection(const RTCFilterFunctionNArguments * args, VolumeStack * volumes)
    {
        Vec3f wi(RTCRayN_dir_x(args->ray, args->N, 0), RTCRayN_dir_y(args->ray, args->N, 0), RTCRayN_dir_z(args->ray, args->N, 0));
        wi.normalize();
        Vec3f n(RTCHitN_Ng_x(args->hit, args->N, 0), RTCHitN_Ng_y(args->hit, args->N, 0), RTCHitN_Ng_z(args->hit, args->N, 0));
        n.normalize();

        if(n.dot(-wi) > 0.f)
            volumes->add_intersect(RTCRayN_tfar(args->ray, args->N, 0), volume);
        else
            volumes->sub_intersect(RTCRayN_tfar(args->ray, args->N, 0), volume);
    }

    const RTCGeometry& get_rtc_geometry() { return geom; }

    std::shared_ptr<Material> material; // This should be const
    const Transform3f obj_to_world;
    const Transform3f world_to_obj;

protected:
    RTCGeometry geom;
    Box3f bbox;
};

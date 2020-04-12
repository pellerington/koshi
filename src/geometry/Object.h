#pragma once

#include <memory>
#include <iostream>
#include <vector>

#include <Scene/Embree.h>
#include <Util/Surface.h>
#include <Util/Ray.h>
#include <Volume/VolumeStack.h>
#include <Lights/Light.h>
class Surface;
class Material;

class Object
{
public:
    Object(const Transform3f &obj_to_world = Transform3f(),
           std::shared_ptr<Light> light = nullptr,
           std::shared_ptr<Material> material = nullptr,
           std::shared_ptr<Volume> volume = nullptr,
           const bool &hide_camera = false)
    : obj_to_world(obj_to_world), world_to_obj(Transform3f::inverse(obj_to_world)),
      light(light), material(material), volume(volume ? std::shared_ptr<Volume>(new Volume(*volume, &world_to_obj)) : nullptr),
      hide_camera(hide_camera) {}

    enum Type
    {
        Mesh,
        Sphere,
        Box,
        LightArea,
        LightSphere,
        LightEnvironment,
        LightCombiner
    };
    virtual Type get_type() = 0;

    void set_id(const uint id) { this->id = id; }
    const RTCGeometry& get_rtc_geometry() { return geom; }

    const Box3f get_bbox() { return bbox; };
    const Transform3f obj_to_world;
    const Transform3f world_to_obj;

    virtual void process_intersection(Surface &surface, const RTCRayHit &rtcRayHit, const Ray &ray)
    {
        surface.set_hit(
            ray.get_position(ray.t),
            Vec3f(rtcRayHit.hit.Ng_x, rtcRayHit.hit.Ng_y, rtcRayHit.hit.Ng_z).normalized(),
            Vec3f(rtcRayHit.hit.Ng_x, rtcRayHit.hit.Ng_y, rtcRayHit.hit.Ng_z).normalized(),
            rtcRayHit.hit.u,
            rtcRayHit.hit.v
        );
    }

    virtual bool use_intersection_filter()
    {
        return hide_camera || volume;
    }

    virtual void process_intersection_visibility(const RTCFilterFunctionNArguments * args)
    {
        IntersectContext * context = (IntersectContext*) args->context;
        bool visible = !(context->ray->camera && hide_camera);
        visible = visible && (material || light); // These should be textured
        *args->valid = visible ? -1 : 0;
    }

    virtual void process_intersection_volume(const RTCFilterFunctionNArguments * args)
    {
        IntersectContext * context = (IntersectContext*) args->context;
        const double t = RTCRayN_tfar(args->ray, args->N, 0);
        const Vec3f normal = Vec3f(RTCHitN_Ng_x(args->hit, args->N, 0), RTCHitN_Ng_y(args->hit, args->N, 0), RTCHitN_Ng_z(args->hit, args->N, 0));
        const bool front = normal.dot(-context->ray->dir) > 0.f;

        if(front)
            context->volumes->add_intersect(t, volume, (material != nullptr));
        else
            context->volumes->sub_intersect(t, volume);
    }

    virtual bool sample_light(const uint num_samples, const Vec3f * pos, const Vec3f * pfar, std::vector<LightSample> &light_samples, Resources &resources) { return false; }
    virtual bool evaluate_light(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample, Resources &resources) { return false; }

    // These should be const
    std::shared_ptr<Light> light;
    std::shared_ptr<Material> material;
    std::shared_ptr<Volume> volume;

protected:

    void set_null_rtc_geometry()
    {
        geom = rtcNewGeometry(Embree::rtc_device, RTC_GEOMETRY_TYPE_TRIANGLE);
        rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4, sizeof(VERT_DATA), 0);
    }

    bool hide_camera;
    RTCGeometry geom;
    Box3f bbox;
    uint id = -1;
};

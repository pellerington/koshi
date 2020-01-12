#pragma once

#include <memory>
#include <iostream>
#include <vector>

#include "../Scene/Embree.h"
#include "../Util/Surface.h"
#include "../Util/Ray.h"
#include "../Volume/VolumeStack.h"
#include "../Lights/Light.h"
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

    const Box3f get_bbox() { return bbox; };
    const Transform3f obj_to_world;
    const Transform3f world_to_obj;

    const RTCGeometry& get_rtc_geometry() { return geom; }

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

    void set_filter_function(void (*_intersection_callback)(const RTCFilterFunctionNArguments *))
    {
        intersection_callback = _intersection_callback;
    }

    virtual bool variable_visibility() { return hide_camera; }
    virtual bool process_visibility_intersection(const bool camera) { return !(camera && hide_camera); }

    virtual void process_volume_intersection(const Ray &ray, const float t, const bool front, VolumeStack * volumes) // GIVE THIS BETTER ARGUMENTS!
    {
        if(front)
            volumes->add_intersect(t, volume, (material != nullptr));
        else
            volumes->sub_intersect(t, volume);
    }

    virtual bool sample_light(const uint num_samples, const Vec3f * pos, const Vec3f * pfar, std::vector<LightSample> &light_samples, RNG &rng) { return false; }
    virtual bool evaluate_light(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample) { return false; }

    // These should be const
    std::shared_ptr<Light> light;
    std::shared_ptr<Material> material;
    std::shared_ptr<Volume> volume;

    void set_id(const uint _id) { id = _id; }

protected:

    void set_null_rtc_geometry()
    {
        geom = rtcNewGeometry(Embree::rtc_device, RTC_GEOMETRY_TYPE_TRIANGLE);
        rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4, sizeof(VERT_DATA), 0);
    }

    const bool hide_camera;
    RTCGeometry geom;
    Box3f bbox;
    uint id = -1;

    // This can be accessed via the embree api.
    void (*intersection_callback)(const RTCFilterFunctionNArguments *) = nullptr;
};

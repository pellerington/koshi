#pragma once

#include <memory>
#include <iostream>
#include <vector>

#include <base/Object.h>

#include <Util/Resources.h>
#include <Util/Surface.h>
#include <intersection/Ray.h>
#include <Lights/Light.h>

class Surface;
class Material;

class Geometry : public Object
{
public:
    Geometry(const Transform3f &obj_to_world = Transform3f(),
            std::shared_ptr<Light> light = nullptr,
            std::shared_ptr<Material> material = nullptr,
            const bool &hide_camera = false)
    : light(light), 
      material(material),
      obj_to_world(obj_to_world), 
      world_to_obj(Transform3f::inverse(obj_to_world)),
      hide_camera(hide_camera) 
    {
    }

    const Box3f& get_bbox() { return bbox; }
    const Transform3f& get_obj_to_world() { return obj_to_world; }
    const Transform3f& get_world_to_obj() { return world_to_obj; }

    // const Vec3f get_opacity(/* intersection */) { return hide_camera ? VEC3F_ZERO : VEC3F_ONES; }

    // virtual bool use_intersection_filter()
    // {
    //     return hide_camera || volume;
    // }

    // virtual void process_intersection_visibility(const RTCFilterFunctionNArguments * args)
    // {
    //     IntersectContext * context = (IntersectContext*) args->context;
    //     bool visible = !(context->ray->camera && hide_camera);
    //     visible = visible && (material || light); // These should be textured
    //     *args->valid = visible ? -1 : 0;
    // }

    // virtual void process_intersection_volume(const RTCFilterFunctionNArguments * args)
    // {
    //     IntersectContext * context = (IntersectContext*) args->context;
    //     const double t = RTCRayN_tfar(args->ray, args->N, 0);
    //     const Vec3f normal = Vec3f(RTCHitN_Ng_x(args->hit, args->N, 0), RTCHitN_Ng_y(args->hit, args->N, 0), RTCHitN_Ng_z(args->hit, args->N, 0));
    //     const bool front = normal.dot(-context->ray->dir) > 0.f;

    //     if(front)
    //         context->volumes->add_intersect(t, volume, (material != nullptr));
    //     else
    //         context->volumes->sub_intersect(t, volume);
    // }

    // These should be an attribute
    virtual bool sample_light(const uint num_samples, const Vec3f * pos, const Vec3f * pfar, std::vector<LightSample> &light_samples, Resources &resources) { return false; }
    virtual bool evaluate_light(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample, Resources &resources) { return false; }

    // These should be an attribute
    std::shared_ptr<Light> light;
    std::shared_ptr<Material> material;

protected:
    Box3f bbox;
    Transform3f obj_to_world;
    Transform3f world_to_obj;

    bool hide_camera;

};

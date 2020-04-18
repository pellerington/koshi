#pragma once

#include <embree/EmbreeGeometry.h>
#include <geometry/GeometryMesh.h>

class EmbreeGeometrySphere : public EmbreeGeometry
{
public:
    EmbreeGeometrySphere(GeometrySphere * sphere)
    {
        geometry = rtcNewGeometry(Embree::rtc_device, RTC_GEOMETRY_TYPE_USER);
        rtcSetGeometryUserPrimitiveCount(geometry, 1);
        rtcSetGeometryBoundsFunction(geometry, bounds_callback, sphere);
        rtcSetGeometryIntersectFunction(geometry, intersect_callback);
    }

    static void bounds_callback(const RTCBoundsFunctionArguments * args)
    {
        GeometrySphere * sphere = (GeometrySphere*)args->geometryUserPtr;
        const Box3f& bbox = sphere->get_bbox();
        args->bounds_o->lower_x = bbox.min().x; args->bounds_o->upper_x = bbox.max().x;
        args->bounds_o->lower_y = bbox.min().y; args->bounds_o->upper_y = bbox.max().y;
        args->bounds_o->lower_z = bbox.min().z; args->bounds_o->upper_z = bbox.max().z;
    }

    static void intersect_callback(const RTCIntersectFunctionNArguments* args)
    {
        GeometrySphere * sphere = (GeometrySphere*)args->geometryUserPtr;
        EmbreeIntersectContext * context = (EmbreeIntersectContext*)args->context;
        args->valid[0] = 0;

        // TODO: Change all this when we switch to an object intersection model.
        const Transform3f& world_to_obj = sphere->get_world_to_obj();
        const Vec3f& center = sphere->get_center();
        const float& radius_sqr = sphere->get_radius_sqr();

        float t0, t1;
        if(!sphere->is_elliptoid())
        {
            const Vec3f v = context->ray->pos - center;
            const float a = context->ray->dir.dot(context->ray->dir);
            const float b = 2.f * v.dot(context->ray->dir);
            const float c = v.dot(v) - radius_sqr;
            const float discriminant = b*b - 4.f*a*c;
            if(discriminant < 0.f) return;
            const float inv_a = 0.5f / a;
            const float sqrt_d = sqrtf(discriminant);
            t0 = (-b - sqrt_d) * inv_a;
            t1 = (-b + sqrt_d) * inv_a;
        }
        else
        {
            const Vec3f ray_pos_object = world_to_obj * context->ray->pos;
            Vec3f ray_dir_object = world_to_obj.multiply(context->ray->dir, false);
            const float inv_obj_dir_len = 1.f / ray_dir_object.length();
            ray_dir_object *= inv_obj_dir_len;

            const float a = ray_dir_object.dot(ray_dir_object);
            const float b = 2.f * ray_pos_object.dot(ray_dir_object);
            const float c = ray_pos_object.dot(ray_pos_object) - 1.f;
            const float discriminant = b*b - 4.f*a*c;
            if(discriminant < 0.f) return;
            const float inv_a = 0.5f / a;
            const float sqrt_d = sqrtf(discriminant);
            t0 = inv_obj_dir_len * (-b - sqrt_d) * inv_a;
            t1 = inv_obj_dir_len * (-b + sqrt_d) * inv_a;
        }

        RTCRayN * rays = RTCRayHitN_RayN(args->rayhit, args->N);
        float& ray_tfar = RTCRayN_tfar(rays, args->N, 0);
        const float ray_tnear = 0.00001f + RTCRayN_tnear(rays, args->N, 0);

        if(t0 < ray_tfar && t0 > ray_tnear)
        {
            const Vec3f sphere_position = context->ray->pos + context->ray->dir * t0;
            const Vec3f normal = (sphere_position - center).normalized();
            args->valid[0] = -1;
            const float tfar_prev = ray_tfar;
            ray_tfar = t0;

            RTCHit potentialhit;
            // potentialhit.u = 0.0f;
            // potentialhit.v = 0.0f;
            potentialhit.Ng_x = normal.x;
            potentialhit.Ng_y = normal.y;
            potentialhit.Ng_z = normal.z;
            potentialhit.geomID = args->geomID;

            RTCFilterFunctionNArguments filter_args;
            filter_args.valid = args->valid;
            filter_args.geometryUserPtr = args->geometryUserPtr;
            filter_args.context = args->context;
            filter_args.ray = rays;
            filter_args.hit = (RTCHitN*)&potentialhit;
            filter_args.N = 1;

            rtcFilterIntersection(args, &filter_args);

            if(!args->valid[0])
                ray_tfar = tfar_prev;
            else
                rtcCopyHitToHitN(RTCRayHitN_HitN(args->rayhit, args->N), &potentialhit, args->N, 0);
        }

        if(t1 < ray_tfar && t1 > ray_tnear)
        {
            const Vec3f sphere_position = context->ray->pos + context->ray->dir * t1;
            const Vec3f normal = (sphere_position - center).normalized();
            args->valid[0] = -1;
            const float tfar_prev = ray_tfar;
            ray_tfar = t1;

            RTCHit potentialhit;
            // potentialhit.u = 0.0f;
            // potentialhit.v = 0.0f;
            potentialhit.Ng_x = normal.x;
            potentialhit.Ng_y = normal.y;
            potentialhit.Ng_z = normal.z;
            potentialhit.geomID = args->geomID;

            RTCFilterFunctionNArguments filter_args;
            filter_args.valid = args->valid;
            filter_args.geometryUserPtr = args->geometryUserPtr;
            filter_args.context = args->context;
            filter_args.ray = rays;
            filter_args.hit = (RTCHitN*)&potentialhit;
            filter_args.N = 1;

            rtcFilterIntersection(args, &filter_args);

            if(!args->valid[0])
                ray_tfar = tfar_prev;
            else
                rtcCopyHitToHitN(RTCRayHitN_HitN(args->rayhit, args->N), &potentialhit, args->N, 0);
        }
    }

};
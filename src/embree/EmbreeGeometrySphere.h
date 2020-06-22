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
        const Box3f& bbox = sphere->get_obj_bbox();
        args->bounds_o->lower_x = bbox.min().x; args->bounds_o->upper_x = bbox.max().x;
        args->bounds_o->lower_y = bbox.min().y; args->bounds_o->upper_y = bbox.max().y;
        args->bounds_o->lower_z = bbox.min().z; args->bounds_o->upper_z = bbox.max().z;
    }

    static void intersect_callback(const RTCIntersectFunctionNArguments* args)
    {
        args->valid[0] = 0;

        RTCRayN * ray = RTCRayHitN_RayN(args->rayhit, 0);
        const Vec3f pos(RTCRayN_org_x(ray, args->N, 0), RTCRayN_org_y(ray, args->N, 0), RTCRayN_org_z(ray, args->N, 0));
        const Vec3f dir(RTCRayN_dir_x(ray, args->N, 0), RTCRayN_dir_y(ray, args->N, 0), RTCRayN_dir_z(ray, args->N, 0));

        float t[2];
        const float a = dir.dot(dir);
        const float b = 2.f * pos.dot(dir);
        const float c = pos.dot(pos) - SPHERE_RADIUS;
        const float discriminant = b*b - 4.f*a*c;
        if(discriminant < 0.f) return;
        const float inv_a = 0.5f / a;
        const float sqrt_d = sqrtf(discriminant);
        t[0] = (-b - sqrt_d) * inv_a;
        t[1] = (-b + sqrt_d) * inv_a;

        float& tfar = RTCRayN_tfar(ray, args->N, 0);
        const float tnear = 1e-5f + RTCRayN_tnear(ray, args->N, 0);

        for(uint i = 0; i < 2; i++)
            if(t[i] < tfar && t[i] > tnear)
            {
                const Vec3f normal = (pos + dir * t[i]).normalized();
                args->valid[0] = -1;
                const float tfar_prev = tfar;
                tfar = t[i];

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
                filter_args.ray = ray;
                filter_args.hit = (RTCHitN*)&potentialhit;
                filter_args.N = 1;

                rtcFilterIntersection(args, &filter_args);

                if(!args->valid[0])
                    tfar = tfar_prev;
                else
                    rtcCopyHitToHitN(RTCRayHitN_HitN(args->rayhit, args->N), &potentialhit, args->N, 0);
            }
    }

};
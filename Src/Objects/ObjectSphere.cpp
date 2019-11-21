#include "ObjectSphere.h"

ObjectSphere::ObjectSphere(const Transform3f &obj_to_world, std::shared_ptr<Material> material, std::shared_ptr<Volume> volume)
: Object(obj_to_world, nullptr, material, volume)
{
    bbox = obj_to_world * Box3f(Vec3f(-1.f), Vec3f(1.f));
    center = obj_to_world * Vec3f(0.f);

    geom = rtcNewGeometry(Embree::rtc_device, RTC_GEOMETRY_TYPE_USER);
    rtcSetGeometryUserPrimitiveCount(geom, 1);
    rtcSetGeometryUserData(geom, this);

    auto bbox_callback = [](const RTCBoundsFunctionArguments * args)
    {
        ObjectSphere * sphere = reinterpret_cast<ObjectSphere*>(args->geometryUserPtr);
        args->bounds_o->lower_x = sphere->bbox.min().x; args->bounds_o->upper_x = sphere->bbox.max().x;
        args->bounds_o->lower_y = sphere->bbox.min().y; args->bounds_o->upper_y = sphere->bbox.max().y;
        args->bounds_o->lower_z = sphere->bbox.min().z; args->bounds_o->upper_z = sphere->bbox.max().z;
    };
    rtcSetGeometryBoundsFunction(geom, bbox_callback, this);

    auto intersect_callback = [](const RTCIntersectFunctionNArguments* args)
    {
        ObjectSphere * sphere = reinterpret_cast<ObjectSphere*>(args->geometryUserPtr);
        RTCRayN * rays = RTCRayHitN_RayN(args->rayhit, args->N);
        RTCHitN * hits = RTCRayHitN_HitN(args->rayhit, args->N);
        for(uint i = 0; i < args->N; i++)
        {
            args->valid[i] = 0;

            const Vec3f ray_pos_world = Vec3f(RTCRayN_org_x(rays, args->N, i), RTCRayN_org_y(rays, args->N, i), RTCRayN_org_z(rays, args->N, i));
            const Vec3f ray_dir_world = Vec3f(RTCRayN_dir_x(rays, args->N, i), RTCRayN_dir_y(rays, args->N, i), RTCRayN_dir_z(rays, args->N, i));
            const Vec3f ray_pos = sphere->world_to_obj * ray_pos_world;
            const Vec3f ray_dir = (sphere->world_to_obj.multiply(ray_dir_world, false)).normalized();

            float t = -1.f;
            const float a = ray_dir.dot(ray_dir);
            const float b = 2.f * ray_pos.dot(ray_dir);
            const float c = ray_pos.dot(ray_pos) - 1.f;
            const float discriminant = b*b - 4.f*a*c;
            if(discriminant >= 0.f)
                t = (-b - sqrtf(discriminant)) / (2.f*a);

            if(t < 0.f)
                continue;

            const Vec3f sphere_position = sphere->obj_to_world * (ray_pos + t * ray_dir);
            t = (sphere_position - ray_pos_world).length();
            if(t < RTCRayN_tfar(rays, args->N, i) && t > RTCRayN_tnear(rays, args->N, i))
            {
                args->valid[i] = -1;
                RTCRayN_tfar(rays, args->N, i) = t;
                const Vec3f normal = (sphere_position - sphere->center).normalized();
                RTCHitN_Ng_x(hits, args->N, i) = normal.x;
                RTCHitN_Ng_y(hits, args->N, i) = normal.y;
                RTCHitN_Ng_z(hits, args->N, i) = normal.z;
                RTCHitN_geomID(hits, args->N, i) = sphere->id;
            }
        }
    };
    rtcSetGeometryIntersectFunction(geom, intersect_callback);
}

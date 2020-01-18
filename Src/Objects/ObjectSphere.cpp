#include "ObjectSphere.h"

ObjectSphere::ObjectSphere(const Transform3f &obj_to_world, std::shared_ptr<Material> material, std::shared_ptr<Volume> volume, std::shared_ptr<Light> light, const bool hide_camera)
: Object(obj_to_world, light, material, volume, hide_camera)
{
    bbox = obj_to_world * Box3f(Vec3f(-1.f), Vec3f(1.f));
    center = obj_to_world * Vec3f(0.f);

    x_len = obj_to_world.multiply(Vec3f(1.f, 0.f, 0.f), false).length();
    y_len = obj_to_world.multiply(Vec3f(0.f, 1.f, 0.f), false).length();
    z_len = obj_to_world.multiply(Vec3f(0.f, 0.f, 1.f), false).length();
    radius = std::max(x_len, std::max(y_len, z_len));
    radius_sqr = radius * radius;
    elliptoid = fabs(x_len - y_len) > 0.01f || fabs(x_len - z_len) > 0.01f || fabs(z_len - y_len) > 0.01f;

    geom = rtcNewGeometry(Embree::rtc_device, RTC_GEOMETRY_TYPE_USER);
    rtcSetGeometryUserPrimitiveCount(geom, 1);

    auto bbox_callback = [](const RTCBoundsFunctionArguments * args)
    {
        ObjectSphere * sphere = (ObjectSphere*)args->geometryUserPtr;
        args->bounds_o->lower_x = sphere->bbox.min().x; args->bounds_o->upper_x = sphere->bbox.max().x;
        args->bounds_o->lower_y = sphere->bbox.min().y; args->bounds_o->upper_y = sphere->bbox.max().y;
        args->bounds_o->lower_z = sphere->bbox.min().z; args->bounds_o->upper_z = sphere->bbox.max().z;
    };
    rtcSetGeometryBoundsFunction(geom, bbox_callback, this);
    rtcSetGeometryIntersectFunction(geom, ObjectSphere::intersect_callback);
}

void ObjectSphere::intersect_callback(const RTCIntersectFunctionNArguments* args)
{
    ObjectSphere * sphere = (ObjectSphere*)args->geometryUserPtr;

    // Move proper scene intersect context. Need to fix cyclc dependcies.
    struct IntersectContext : public RTCIntersectContext { Ray * ray; };
    IntersectContext * context = (IntersectContext*) args->context;
    if(sphere->hide_camera && context->ray->camera) return;

    args->valid[0] = 0;

    float t0, t1;
    if(!sphere->elliptoid)
    {
        const Vec3f v = context->ray->pos - sphere->center;
        const float a = context->ray->dir.dot(context->ray->dir);
        const float b = 2.f * v.dot(context->ray->dir);
        const float c = v.dot(v) - sphere->radius_sqr;
        const float discriminant = b*b - 4.f*a*c;
        if(discriminant < 0.f) return;
        const float inv_a = 0.5f / a;
        const float sqrt_d = sqrtf(discriminant);
        t0 = (-b - sqrt_d) * inv_a;
        t1 = (-b + sqrt_d) * inv_a;
    }
    else
    {
        const Vec3f ray_pos_object = sphere->world_to_obj * context->ray->pos;
        Vec3f ray_dir_object = sphere->world_to_obj.multiply(context->ray->dir, false);
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
        const Vec3f normal = (sphere_position - sphere->center).normalized();
        args->valid[0] = -1;
        const float tfar_prev = ray_tfar;
        ray_tfar = t0;

        RTCHit potentialhit;
        // potentialhit.u = 0.0f;
        // potentialhit.v = 0.0f;
        potentialhit.Ng_x = normal.x;
        potentialhit.Ng_y = normal.y;
        potentialhit.Ng_z = normal.z;
        potentialhit.geomID = sphere->id;

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
        const Vec3f normal = (sphere_position - sphere->center).normalized();
        args->valid[0] = -1;
        const float tfar_prev = ray_tfar;
        ray_tfar = t1;

        RTCHit potentialhit;
        // potentialhit.u = 0.0f;
        // potentialhit.v = 0.0f;
        potentialhit.Ng_x = normal.x;
        potentialhit.Ng_y = normal.y;
        potentialhit.Ng_z = normal.z;
        potentialhit.geomID = sphere->id;

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

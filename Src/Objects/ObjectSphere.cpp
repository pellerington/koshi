#include "ObjectSphere.h"

ObjectSphere::ObjectSphere(const Vec3f &position, const float &scale, std::shared_ptr<Material> material)
: Object(material), position(position), scale(scale), scale_sqr(scale*scale)
{
    bbox = Box3f(position-Vec3f(scale), position+Vec3f(scale));

    mesh = rtcNewGeometry(Embree::rtc_device, RTC_GEOMETRY_TYPE_SPHERE_POINT);

    RTCVertex * vertex = (RTCVertex*) rtcSetNewGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4, sizeof(RTCVertex), 1);
    vertex[0].x = position.x;
    vertex[0].y = position.y;
    vertex[0].z = position.z;
    vertex[0].r = scale;
}

Surface ObjectSphere::process_intersection(const RTCRayHit &rtcRayHit, const Ray &ray)
{
    return Surface (
        this,
        ray.get_position(ray.t),
        Vec3f(rtcRayHit.hit.Ng_x, rtcRayHit.hit.Ng_y, rtcRayHit.hit.Ng_z).normalized(),
        ray.dir,
        rtcRayHit.hit.u, // These are incorrect for spheres and need to be explicitly set
        rtcRayHit.hit.v
    );
}

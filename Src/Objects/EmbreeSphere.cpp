#include "Sphere.h"

#if EMBREE

Sphere::Sphere(Vec3f position, float scale, std::shared_ptr<Material> material)
: Object(material), position(position), scale(scale), scale_sqr(scale*scale)
{
    bbox = Eigen::AlignedBox3f(position-Vec3f(scale, scale, scale), position+Vec3f(scale, scale, scale));

    mesh = rtcNewGeometry(Embree::rtc_device, RTC_GEOMETRY_TYPE_SPHERE_POINT);

    RTCVertex * vertex = (RTCVertex*) rtcSetNewGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4, sizeof(RTCVertex), 1);
    vertex[0].x = position.x();
    vertex[0].y = position.y();
    vertex[0].z = position.z();
    vertex[0].r = scale;

}

void Sphere::process_intersection(RTCRayHit &rtcRayHit, Ray &ray, Surface &surface)
{
    surface.position = ray.o + ray.t * ray.dir;
    surface.wi = ray.dir;
    surface.normal = Vec3f(rtcRayHit.hit.Ng_x, rtcRayHit.hit.Ng_y, rtcRayHit.hit.Ng_z);
    surface.normal.normalize();
    surface.enter = surface.normal.dot(ray.dir) < 0;
    surface.u = rtcRayHit.hit.u; // THESE NEED TO BE EXPLICTLY SET !
    surface.v = rtcRayHit.hit.v;
    surface.object = this;
}

bool Sphere::intersect(Ray &ray, Surface &surface)
{
    return false;
}

#endif

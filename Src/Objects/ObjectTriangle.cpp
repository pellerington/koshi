#include "ObjectTriangle.h"

ObjectTriangle::ObjectTriangle(const Vec3f &v0, const Vec3f &v1, const Vec3f &v2, std::shared_ptr<Material> material)
: Object(material),
  vertices({std::make_shared<Vec3f>(v0), std::make_shared<Vec3f>(v1), std::make_shared<Vec3f>(v2)}),
  normals({nullptr, nullptr, nullptr})
{
    init();
}


ObjectTriangle::ObjectTriangle(const Vec3f &v0, const Vec3f &v1, const Vec3f &v2, const Vec3f &n0, const Vec3f &n1, const Vec3f &n2, std::shared_ptr<Material> material)
: Object(material),
  vertices({std::make_shared<Vec3f>(v0), std::make_shared<Vec3f>(v1), std::make_shared<Vec3f>(v2)}),
  normals({std::make_shared<Vec3f>(n0), std::make_shared<Vec3f>(n1), std::make_shared<Vec3f>(n2)})
{
    init();
}

void ObjectTriangle::init()
{
    bbox = Box3f(Vec3f::min(*vertices[0], Vec3f::min(*vertices[1], *vertices[2])),
                 Vec3f::max(*vertices[0], Vec3f::max(*vertices[1], *vertices[2])));
    smooth_normals = normals[0] && normals[1] && normals[2];

    mesh = rtcNewGeometry(Embree::rtc_device, RTC_GEOMETRY_TYPE_TRIANGLE);

    RTCVertex * rtc_vertices = (RTCVertex*) rtcSetNewGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(RTCVertex), 3);
    for(uint i = 0; i < 3; i++)
    {
        rtc_vertices[i].x = vertices[i]->x;
        rtc_vertices[i].y = vertices[i]->y;
        rtc_vertices[i].z = vertices[i]->z;
    }

    RTCTriangle * triangles = (RTCTriangle*) rtcSetNewGeometryBuffer(mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(RTCTriangle), 1);
    triangles[0].v0 = 0;
    triangles[0].v1 = 1;
    triangles[0].v2 = 2;
}

Surface ObjectTriangle::process_intersection(const RTCRayHit &rtcRayHit, const Ray &ray)
{
    return Surface (
        this,
        ray.get_position(ray.t),
        Vec3f(rtcRayHit.hit.Ng_x, rtcRayHit.hit.Ng_y, rtcRayHit.hit.Ng_z).normalized(),
        // Smooth normals here (use ray.hit.primID to get triangle)
        // surface.normal = (smooth_normals) ? (1.f - u - v) * *normals[0] + u * *normals[1] + v * *normals[2] : normal;
        ray.dir,
        rtcRayHit.hit.u,
        rtcRayHit.hit.v
    );
}

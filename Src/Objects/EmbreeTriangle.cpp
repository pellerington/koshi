#include "Triangle.h"

#if EMBREE

Triangle::Triangle(std::shared_ptr<Vec3f> v0, std::shared_ptr<Vec3f> v1, std::shared_ptr<Vec3f> v2,
                   std::shared_ptr<Vec3f> n0, std::shared_ptr<Vec3f> n1, std::shared_ptr<Vec3f> n2,
                   std::shared_ptr<Material> material)
: Object(material), vertices({v0, v1, v2}), normals({n0, n1, n2})
{
    init();
}

Triangle::Triangle(const Vec3f &v0, const Vec3f &v1, const Vec3f &v2, std::shared_ptr<Material> material)
: Object(material),
  vertices({std::make_shared<Vec3f>(v0), std::make_shared<Vec3f>(v1), std::make_shared<Vec3f>(v2)}),
  normals({nullptr, nullptr, nullptr})
{
    init();
}


Triangle::Triangle(const Vec3f &v0, const Vec3f &v1, const Vec3f &v2, const Vec3f &n0, const Vec3f &n1, const Vec3f &n2, std::shared_ptr<Material> material)
: Object(material),
  vertices({std::make_shared<Vec3f>(v0), std::make_shared<Vec3f>(v1), std::make_shared<Vec3f>(v2)}),
  normals({std::make_shared<Vec3f>(n0), std::make_shared<Vec3f>(n1), std::make_shared<Vec3f>(n2)})
{
    init();
}

void Triangle::init()
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

void Triangle::process_intersection(const RTCRayHit &rtcRayHit, Ray &ray, Surface &surface)
{
    surface.position = ray.o + ray.t * ray.dir;
    surface.wi = ray.dir;
    surface.normal = Vec3f(rtcRayHit.hit.Ng_x, rtcRayHit.hit.Ng_y, rtcRayHit.hit.Ng_z);
    surface.normal.normalize();
    surface.enter = surface.normal.dot(ray.dir) < 0;
    surface.u = rtcRayHit.hit.u;
    surface.v = rtcRayHit.hit.v;
    //Smooth normals here (use ray.hit.primID to get triangle)
    // surface.normal = (smooth_normals) ? (1.f - u - v) * *normals[0] + u * *normals[1] + v * *normals[2] : normal;
    surface.object = this;
}


bool Triangle::intersect(Ray &ray, Surface &surface)
{
    return false;
}

#endif

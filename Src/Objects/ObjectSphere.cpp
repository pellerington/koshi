#include "ObjectSphere.h"

#include "../Math/RNG.h"

static const float icosahedron_size = 0.44f;
static const float t0 = icosahedron_size * 1.f;
static const float t1 = icosahedron_size * (1.f + sqrtf(5.f)) * 0.5f;

static const Vec3f icosahedron_vertices[12] = {
    Vec3f(-t0,  t1, 0.f), Vec3f( t0,  t1, 0.f),
    Vec3f(-t0, -t1, 0.f), Vec3f( t0, -t1, 0.f),
    Vec3f(0.f, -t0,  t1), Vec3f(0.f,  t0,  t1),
    Vec3f(0.f, -t0, -t1), Vec3f(0.f,  t0, -t1),
    Vec3f( t1, 0.f, -t0), Vec3f( t1, 0.f,  t0),
    Vec3f(-t1, 0.f, -t0), Vec3f(-t1, 0.f,  t0)
};

static const uint icosahedron_indices[60] = {
    0,  11, 5,      0,  5,  1,      0,  1,  7,      0,  7,  10,
    0,  10, 11,     1,  5,  9,      5,  11, 4,      11, 10, 2,
    10, 7,  6,      7,  1,  8,      3,  9,  4,      3,  4,  2,
    3,  2,  6,      3,  6,  8,      3,  8,  9,      4,  9,  5,
    2,  4,  11,     6,  2,  10,     8,  6,  7,      9,  8,  1
};

ObjectSphere::ObjectSphere(std::shared_ptr<Material> material, const Transform3f &obj_to_world, std::shared_ptr<VolumeProperties> volume)
: Object(material, obj_to_world, volume)
{
    bbox = obj_to_world * BOX3F_UNIT;
    position = obj_to_world * Vec3f(0.f);
    scale = obj_to_world.multiply(Vec3f(1.f,0.f,0.f), false).length();

    // geom = rtcNewGeometry(Embree::rtc_device, RTC_GEOMETRY_TYPE_SPHERE_POINT);
    // RTCVertex * vertex = (RTCVertex*) rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT4, sizeof(RTCVertex), 1);
    // vertex[0].x = position.x;
    // vertex[0].y = position.y;
    // vertex[0].z = position.z;
    // vertex[0].r = scale;

    geom = rtcNewGeometry(Embree::rtc_device, RTC_GEOMETRY_TYPE_SUBDIVISION);

    RTCVertex * vertices = (RTCVertex*) rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(RTCVertex), 12);
    for(uint i = 0; i < 12; i++)
    {
        const Vec3f v = obj_to_world * icosahedron_vertices[i];
        vertices[i].x = v.x; vertices[i].y = v.y; vertices[i].z = v.z;
    }

    uint * indices = (uint*) rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT, sizeof(uint), 60);
    for(uint i = 0; i < 60; i++)
        indices[i] = icosahedron_indices[i];

    uint * faces = (uint*) rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_FACE, 0, RTC_FORMAT_UINT, sizeof(uint), 20);
    for(uint i = 0; i < 20; i++)
        faces[i] = 3;

    rtcSetGeometryTessellationRate(geom, 8.f);
}

Surface ObjectSphere::process_intersection(const RTCRayHit &rtcRayHit, const Ray &ray)
{
    const Vec3f hit_point = ray.get_position(ray.t);
    return Surface (
        this,
        hit_point,
        (hit_point - position).normalized(), /*Vec3f(rtcRayHit.hit.Ng_x, rtcRayHit.hit.Ng_y, rtcRayHit.hit.Ng_z).normalized(),*/
        ray.dir,
        rtcRayHit.hit.u, // These are incorrect for spheres and need to be explicitly set
        rtcRayHit.hit.v
    );
}

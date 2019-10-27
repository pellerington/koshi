#include "ObjectBox.h"

#include "../Math/RNG.h"

static Vec3f box_vertices[8] = {
    Vec3f(0.f, 0.f, 0.f),
    Vec3f(1.f, 0.f, 0.f),
    Vec3f(1.f, 0.f, 1.f),
    Vec3f(0.f, 0.f, 1.f),
    Vec3f(0.f, 1.f, 0.f),
    Vec3f(1.f, 1.f, 0.f),
    Vec3f(1.f, 1.f, 1.f),
    Vec3f(0.f, 1.f, 1.f)
};

static uint box_indices[6][4] = {
    {0, 4, 5, 1},
    {1, 5, 6, 2},
    {2, 6, 7, 3},
    {0, 3, 7, 4},
    {4, 7, 6, 5},
    {0, 1, 2, 3},
};

ObjectBox::ObjectBox(std::shared_ptr<Material> material, const Transform3f &obj_to_world, std::shared_ptr<Volume> volume)
: Object(material, obj_to_world, volume)
{
    bbox = obj_to_world * BOX3F_UNIT;

    geom = rtcNewGeometry(Embree::rtc_device, RTC_GEOMETRY_TYPE_QUAD);

    VERT_DATA * vertices = (VERT_DATA*) rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(VERT_DATA), 8);
    for(uint i = 0; i < 8; i++)
    {
        const Vec3f v = obj_to_world * box_vertices[i];
        vertices[i].x = v.x; vertices[i].y = v.y; vertices[i].z = v.z;
    }

    QUAD_DATA * quads = (QUAD_DATA*) rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT4, sizeof(QUAD_DATA), 6);
    for(uint i = 0; i < 6; i++)
    {
        quads[i].v0 = box_indices[i][0]; quads[i].v1 = box_indices[i][1];
        quads[i].v2 = box_indices[i][2]; quads[i].v3 = box_indices[i][3];
    }
}

Surface ObjectBox::process_intersection(const RTCRayHit &rtcRayHit, const Ray &ray)
{
    return Surface(
        ray.get_position(ray.t),
        Vec3f(rtcRayHit.hit.Ng_x, rtcRayHit.hit.Ng_y, rtcRayHit.hit.Ng_z).normalized(),
        Vec3f(rtcRayHit.hit.Ng_x, rtcRayHit.hit.Ng_y, rtcRayHit.hit.Ng_z).normalized(),
        ray.dir,
        rtcRayHit.hit.u,
        rtcRayHit.hit.v,
        ray.ior
    );
}

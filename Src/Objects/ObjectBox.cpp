#include "ObjectBox.h"

#include "../Math/RNG.h"

ObjectBox::ObjectBox(std::shared_ptr<Material> material, const Transform3f &obj_to_world)
: Object(material, obj_to_world)
{
    bbox = obj_to_world * BOX3F_UNIT;

    mesh = rtcNewGeometry(Embree::rtc_device, RTC_GEOMETRY_TYPE_QUAD);

    RTCVertex * vertices = (RTCVertex*) rtcSetNewGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(RTCVertex), 8);
    uint i = 0;
    for(float x = 0.f; x <= 1.f; x++)
    for(float y = 0.f; y <= 1.f; y++)
    for(float z = 0.f; z <= 1.f; z++, i++)
    {
        const Vec3f v = obj_to_world * Vec3f(x, y, z, 1);
        vertices[i].x = v.x; vertices[i].y = v.y; vertices[i].z = v.z;
    }

    RTCQuad * quads = (RTCQuad*) rtcSetNewGeometryBuffer(mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT4, sizeof(RTCQuad), 6);
    i = 0;
    quads[i].v0 = 0; quads[i].v1 = 1; quads[i].v2 = 3; quads[i].v3 = 2; i++;
    quads[i].v0 = 4; quads[i].v1 = 6; quads[i].v2 = 7; quads[i].v3 = 5; i++;
    quads[i].v0 = 0; quads[i].v1 = 4; quads[i].v2 = 5; quads[i].v3 = 1; i++;
    quads[i].v0 = 2; quads[i].v1 = 3; quads[i].v2 = 7; quads[i].v3 = 6; i++;
    quads[i].v0 = 0; quads[i].v1 = 2; quads[i].v2 = 6; quads[i].v3 = 4; i++;
    quads[i].v0 = 1; quads[i].v1 = 5; quads[i].v2 = 7; quads[i].v3 = 3; i++;
}

Surface ObjectBox::process_intersection(const RTCRayHit &rtcRayHit, const Ray &ray)
{
    return Surface(
        this,
        ray.get_position(ray.t),
        Vec3f(rtcRayHit.hit.Ng_x, rtcRayHit.hit.Ng_y, rtcRayHit.hit.Ng_z).normalized(),
        ray.dir,
        rtcRayHit.hit.u,
        rtcRayHit.hit.v
    );
}

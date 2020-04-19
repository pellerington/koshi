#pragma once

#include <embree/EmbreeGeometry.h>
#include <geometry/GeometryArea.h>

class EmbreeGeometryArea : public EmbreeGeometry
{
public:
    EmbreeGeometryArea(GeometryArea * area)
    {
        geometry = rtcNewGeometry(Embree::rtc_device, RTC_GEOMETRY_TYPE_QUAD);

        const Transform3f& obj_to_world = area->get_obj_to_world();

        VERT_DATA * vertices = (VERT_DATA*) rtcSetNewGeometryBuffer(
            geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(VERT_DATA), 4);
        for(uint i = 0; i < 4; i++)
        {
            const Vec3f v = obj_to_world * GeometryArea::vertices[i];
            vertices[i].x = v.x; vertices[i].y = v.y; vertices[i].z = v.z;
        }

        QUAD_DATA * quad = (QUAD_DATA*) rtcSetNewGeometryBuffer(
            geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT4, sizeof(QUAD_DATA), 1);
        quad[0].v0 = 0; quad[0].v1 = 1; quad[0].v2 = 2; quad[0].v3 = 3;
    }
};
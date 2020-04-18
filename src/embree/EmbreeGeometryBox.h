#pragma once

#include <embree/EmbreeGeometry.h>
#include <geometry/GeometryBox.h>

class EmbreeGeometryBox : public EmbreeGeometry
{
public:
    EmbreeGeometryBox(GeometryBox * box)
    {
        geometry = rtcNewGeometry(Embree::rtc_device, RTC_GEOMETRY_TYPE_QUAD);

        const Transform3f& obj_to_world = box->get_obj_to_world();

        VERT_DATA * vertices = (VERT_DATA*) rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(VERT_DATA), 8);
        for(uint i = 0; i < 8; i++)
        {
            const Vec3f v = obj_to_world * GeometryBox::vertices[i];
            vertices[i].x = v.x; vertices[i].y = v.y; vertices[i].z = v.z;
        }

        QUAD_DATA * quads = (QUAD_DATA*) rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT4, sizeof(QUAD_DATA), 6);
        for(uint i = 0; i < 6; i++)
        {
            quads[i].v0 = GeometryBox::indices[i][0]; quads[i].v1 = GeometryBox::indices[i][1];
            quads[i].v2 = GeometryBox::indices[i][2]; quads[i].v3 = GeometryBox::indices[i][3];
        }
    }
};
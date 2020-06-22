#pragma once

#include <embree/EmbreeGeometry.h>
#include <geometry/GeometryBox.h>

class EmbreeGeometryBox : public EmbreeGeometry
{
public:
    EmbreeGeometryBox(GeometryBox * box)
    {
        geometry = rtcNewGeometry(Embree::rtc_device, RTC_GEOMETRY_TYPE_QUAD);
        rtcSetSharedGeometryBuffer(
            geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, vertices, 0, sizeof(float[4]), 8);
        rtcSetSharedGeometryBuffer(
            geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT4, indices, 0, sizeof(uint[4]), 6);
    }

private:
    static const float vertices[8][4];
    static const uint indices[6][4];
};
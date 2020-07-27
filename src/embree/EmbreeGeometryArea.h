#pragma once

#include <koshi/embree/EmbreeGeometry.h>
#include <koshi/geometry/GeometryArea.h>

class EmbreeGeometryArea : public EmbreeGeometry
{
public:
    EmbreeGeometryArea(GeometryArea * area)
    {
        geometry = rtcNewGeometry(Embree::rtc_device, RTC_GEOMETRY_TYPE_QUAD);
        rtcSetSharedGeometryBuffer(
            geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, vertices, 0, sizeof(float[4]), 4);
        rtcSetSharedGeometryBuffer(
            geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT4, indices, 0, sizeof(uint[4]), 1);
    }

private:
    static const float vertices[4][4];
    static const uint indices[1][4];
};
#pragma once

#include <embree/EmbreeGeometry.h>
#include <geometry/GeometryMesh.h>

class EmbreeGeometryMesh : public EmbreeGeometry
{
public:
    EmbreeGeometryMesh(GeometryMesh * mesh)
    {
        geometry = rtcNewGeometry(Embree::rtc_device, RTC_GEOMETRY_TYPE_TRIANGLE);

        rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 
                                    0, RTC_FORMAT_FLOAT3, mesh->get_vertices(), 
                                    0, sizeof(VERT_DATA), mesh->get_vertices_size());
        rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 
                                    0, RTC_FORMAT_UINT3, mesh->get_indicies(), 
                                    0, sizeof(TRI_DATA), mesh->get_triangles_size());
    }
};
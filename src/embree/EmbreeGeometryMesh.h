#pragma once

#include <embree/EmbreeGeometry.h>
#include <geometry/GeometryMesh.h>

class EmbreeGeometryMesh : public EmbreeGeometry
{
public:
    EmbreeGeometryMesh(GeometryMesh * mesh)
    {
        geometry = rtcNewGeometry(Embree::rtc_device, RTC_GEOMETRY_TYPE_TRIANGLE);

        const GeometryMeshAttribute * vertices = mesh->get_mesh_attribute("vertices");
        // if ( !vertices ) do something.

        rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 
                                    0, RTC_FORMAT_FLOAT3, vertices->array, 
                                    0, sizeof(float4), vertices->array_size);
        rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 
                                    0, RTC_FORMAT_UINT3, vertices->indices, 
                                    0, sizeof(uint3), vertices->indices_size);
    }
};
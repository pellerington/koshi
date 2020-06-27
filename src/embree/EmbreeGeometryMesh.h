#pragma once

#include <embree/EmbreeGeometry.h>
#include <geometry/GeometryMesh.h>

class EmbreeGeometryMesh : public EmbreeGeometry
{
public:
    EmbreeGeometryMesh(GeometryMesh * mesh)
    {
        geometry = rtcNewGeometry(Embree::rtc_device, RTC_GEOMETRY_TYPE_TRIANGLE);

        const GeometryMeshAttribute * vertices = dynamic_cast<const GeometryMeshAttribute *>(mesh->get_geometry_attribute("vertices"));
        if (!vertices)
            return;

        rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 
                                   0, RTC_FORMAT_FLOAT3, vertices->array, 
                                   0, sizeof(float[vertices->array_item_size]), 
                                   vertices->array_item_count);
        rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 
                                   0, RTC_FORMAT_UINT3, vertices->indices, 
                                   0, sizeof(uint[vertices->indices_item_size]), 
                                   vertices->indices_item_count);
    }
};
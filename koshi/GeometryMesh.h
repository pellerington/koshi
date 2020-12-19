#pragma once

#include <koshi/Geometry.h>
#include <koshi/Format.h>

// TODO: Remove this limitation. 
#define MAX_MESH_ATTRIBUTES 16

KOSHI_OPEN_NAMESPACE

struct GeometryMeshAttribute {
    std::string name;
    Format format;
    const void * data;
    uint data_stride; // 3 for vertices/normals. 2 for uvs.
    uint data_size;
    // Type type = PerVertices / PerFace
    const uint32_t * indices;
    uint indices_size;
    // uint indices_stride; QUADS OR TRIANGLES?
};

class GeometryMesh : public Geometry
{
public:
    GeometryMesh() : attributes_size(0) {}

    void setAttribute(const std::string& name, const Format& format, const uint& data_size, const uint& data_stride, const void * data, const uint& indices_size, const uint * indices)
    {
        auto setAttribute = [&](const uint& i)
        {
            attributes[i].name = name;
            attributes[i].format = format;
            attributes[i].data = data;
            attributes[i].data_stride = data_stride;
            attributes[i].data_size = data_size;
            attributes[i].indices = indices;
            attributes[i].indices_size = indices_size;
        };

        for(uint i = 0; i < attributes_size; i++)
            if(attributes[i].name == name)
                setAttribute(i);

        if(attributes_size == MAX_MESH_ATTRIBUTES)
            return; // ERROR HERE
        
        setAttribute(attributes_size);
        attributes_size++;
    }

    GeometryMeshAttribute * getAttribute(const std::string& name)
    {
        for(uint i = 0; i < attributes_size; i++)
            if(attributes[i].name == name)
                return &attributes[i];
        return nullptr;
    }

private:
    uint attributes_size;
    GeometryMeshAttribute attributes[16];
};

KOSHI_CLOSE_NAMESPACE
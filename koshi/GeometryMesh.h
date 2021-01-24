#pragma once

#include <cuda_runtime.h>

#include <koshi/Geometry.h>
#include <koshi/Format.h>
#include <koshi/OptixHelpers.h>

// TODO: Remove this limitation. 
#define MAX_MESH_ATTRIBUTES 16
#define MAX_ATTRIBUTE_NAME_LENGTH 64u

KOSHI_OPEN_NAMESPACE

struct GeometryMeshAttribute {
    class AttributeName 
    {
    public:
        AttributeName() {}

        AttributeName(const std::string& name)
        {
            uint size = std::min(MAX_ATTRIBUTE_NAME_LENGTH, (uint)name.size());
            name.copy(data, size);
            data[size] = '\0';
        }

        DEVICE_FUNCTION bool operator==(const char * name)
        {
            for(uint i = 0; i < MAX_ATTRIBUTE_NAME_LENGTH; i++)
            {
                if(name[i] != data[i]) return false;
                if(name[i] == '\0') return true;
            }
            return true;
        }

        bool operator==(const std::string& name)
        {
            return std::string(data) == name;
        }

    private:
        char data[MAX_ATTRIBUTE_NAME_LENGTH+1];
        
    } name;

    Format format;
    // Type type = PerVertices / PerFace

    const void * data;
    void * d_data;
    // size of data = data_stride * data_size?
    uint data_size;
    uint data_stride; // 3 for vertices/normals. 2 for uvs.

    const uint32_t * indices;
    uint32_t * d_indices;
    uint indices_size;
    uint indices_stride;

    // OWNERESHIP?
};

class GeometryMesh : public Geometry
{
public:
    GeometryMesh() : attributes_size(0) {}
    ~GeometryMesh();

    GeometryType type() { return GeometryType::MESH; }

    void setAttribute(const std::string& name, const Format& format, const uint& data_size, const uint& data_stride, const void * data, const uint& indices_size, const uint& indices_stride, const uint * indices);

    DEVICE_FUNCTION GeometryMeshAttribute * getAttribute(const char * name)
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
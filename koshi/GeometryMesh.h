#pragma once

#include <cuda_runtime.h>
#include <algorithm>

#include <koshi/Vec4.h>
#include <koshi/Geometry.h>
#include <koshi/Format.h>
#include <koshi/OptixHelpers.h>
#include <koshi/Intersect.h>

// TODO: Remove this limitation. 
#define MAX_GEOMETRY_MESH_ATTRIBUTES 16
#define MAX_GEOMETRY_MESH_ATTRIBUTE_NAME_LENGTH 64u

KOSHI_OPEN_NAMESPACE

struct GeometryMeshAttribute {
    class AttributeName 
    {
    public:
        AttributeName() {}
        AttributeName(const std::string& name)
        {
            uint size = std::min(MAX_GEOMETRY_MESH_ATTRIBUTE_NAME_LENGTH, (uint)name.size());
            name.copy(data, size);
            data[size] = '\0';
        }
        DEVICE_FUNCTION bool operator==(const char * name)
        {
            for(uint i = 0; i < MAX_GEOMETRY_MESH_ATTRIBUTE_NAME_LENGTH; i++)
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
        char data[MAX_GEOMETRY_MESH_ATTRIBUTE_NAME_LENGTH+1];
        
    } name;

    Format format;
    enum Type { NONE, CONSTANT, UNIFORM, VERTICES, FACE  } type;
    // Ownership

    const void * data;
    void * d_data;
    // size of data = data_stride * data_size?
    uint data_size;
    uint data_stride; // 3 for vertices/normals. 2 for uvs.

    const uint32_t * indices;
    uint32_t * d_indices;
    uint indices_size;
    uint indices_stride;

    DEVICE_FUNCTION Vec4f evaluate(const Intersect& intersect)
    {
        Vec4f out;
        uint size = (data_stride < 4u) ? data_stride : 4u;
        switch (type)
        {
        case CONSTANT: {
            for(uint i = 0; i < size; i++)
                out[i] = ((float*)d_data)[i];
            return out;
        }
        // case UNIFORM:
        //     // Is this how uniform works???
        //     for(uint i = 0; i < size; i++)
        //         out[i] = ((float*)d_data)[intersect.prim*data_stride + i];
        //     return out;
        case VERTICES: {
            const uint32_t p0 = d_indices[intersect.prim*indices_stride+0]*data_stride;
            const uint32_t p1 = d_indices[intersect.prim*indices_stride+1]*data_stride;
            const uint32_t p2 = d_indices[intersect.prim*indices_stride+2]*data_stride;
            for(uint i = 0; i < size; i++)
                out[i] = ((float*)d_data)[p0+i] * (1.f - intersect.uvw.u - intersect.uvw.v) + ((float*)d_data)[p1+i] * intersect.uvw.u + ((float*)d_data)[p2+i] * intersect.uvw.v;
            return out;
        }
        default:
            return out;
        }
    }
};

class GeometryMesh : public Geometry
{
public:
    GeometryMesh() : num_attributes(0), vertices_attribute(MAX_GEOMETRY_MESH_ATTRIBUTES) {}
    ~GeometryMesh();

    // TODO: This should be a variable set in the base class and returned by the base class. (So it can be trivially inlined)
    GeometryType type() { return GeometryType::MESH; }

    void setAttribute(const std::string& name, const Format& format, const GeometryMeshAttribute::Type& type, const uint& data_size, const uint& data_stride, const void * data, const uint& indices_size, const uint& indices_stride, const uint * indices);

    void setVerticesAttribute(const std::string& name)
    {
        for(uint i = 0; i < num_attributes; i++)
            if(attributes[i].name == name)
                vertices_attribute = i;
    }

    DEVICE_FUNCTION bool hasVerticesAttribute() { return vertices_attribute < num_attributes; }
    DEVICE_FUNCTION GeometryMeshAttribute * getVerticesAttribute() { return &attributes[vertices_attribute]; }

    DEVICE_FUNCTION GeometryMeshAttribute * getAttribute(const char * name)
    {
        for(uint i = 0; i < num_attributes; i++)
            if(attributes[i].name == name)
                return &attributes[i];
        return nullptr;
    }

private:
    uint num_attributes;
    GeometryMeshAttribute attributes[MAX_GEOMETRY_MESH_ATTRIBUTES];
    uint vertices_attribute;
};

KOSHI_CLOSE_NAMESPACE
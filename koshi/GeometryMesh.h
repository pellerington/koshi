#pragma once

#include <cuda_runtime.h>

#include <koshi/Geometry.h>
#include <koshi/Format.h>
#include <koshi/OptixHelpers.h>

// TODO: Remove this limitation. 
#define MAX_MESH_ATTRIBUTES 16

KOSHI_OPEN_NAMESPACE

struct GeometryMeshAttribute {
    std::string name;
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

    ~GeometryMesh()
    {
        for(uint i = 0; i < attributes_size; i++)
        {
            CUDA_CHECK(cudaFree(attributes[i].d_data));
            CUDA_CHECK(cudaFree(attributes[i].d_indices));
        }
    }

    void setAttribute(const std::string& name, const Format& format, const uint& data_size, const uint& data_stride, const void * data, const uint& indices_size, const uint& indices_stride, const uint * indices)
    {
        auto setAttribute = [&](const uint& i)
        {
            attributes[i].name = name;
            attributes[i].format = format;
            attributes[i].data = data;
            attributes[i].data_size = data_size;
            attributes[i].data_stride = data_stride;
            attributes[i].indices = indices;
            attributes[i].indices_size = indices_size;
            attributes[i].indices_stride = indices_stride;
    
            CUDA_CHECK(cudaMalloc(&attributes[i].d_data, sizeofFormat(format)*data_size*data_stride));
            CUDA_CHECK(cudaMemcpy(attributes[i].d_data, attributes[i].data, sizeofFormat(format)*data_size*data_stride, cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaMalloc(&attributes[i].d_indices, sizeof(uint32_t)*indices_size*indices_stride));
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(attributes[i].d_indices), attributes[i].indices, sizeof(uint32_t)*indices_size*indices_stride, cudaMemcpyHostToDevice));
        };

        for(uint i = 0; i < attributes_size; i++)
            if(attributes[i].name == name)
            {
                CUDA_CHECK(cudaFree(attributes[i].d_data));
                CUDA_CHECK(cudaFree(attributes[i].d_indices));
                setAttribute(i);
                return;
            }

        if(attributes_size == MAX_MESH_ATTRIBUTES)
            return; // ERROR HERE
        
        setAttribute(attributes_size);
        attributes_size++;

        // TODO: SET DIRTY HERE...
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
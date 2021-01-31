#include <koshi/GeometryMesh.h>

KOSHI_OPEN_NAMESPACE

GeometryMesh::~GeometryMesh()
{
    for(uint i = 0; i < attributes_size; i++)
    {
        CUDA_CHECK(cudaFree(attributes[i].d_data));
        CUDA_CHECK(cudaFree(attributes[i].d_indices));
    }
}

void GeometryMesh::setAttribute(const std::string& name, const Format& format, const GeometryMeshAttribute::Type& type, const uint& data_size, const uint& data_stride, const void * data, const uint& indices_size, const uint& indices_stride, const uint * indices)
{
    auto setAttribute = [&](const uint& i)
    {
        attributes[i].name = name;
        attributes[i].format = format;
        attributes[i].type = type;
        attributes[i].data = data;
        attributes[i].data_size = data_size;
        attributes[i].data_stride = data_stride;
        attributes[i].indices = indices;
        attributes[i].indices_size = indices_size;
        attributes[i].indices_stride = indices_stride;

        // TODO: These should be device agnostic.
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

KOSHI_CLOSE_NAMESPACE
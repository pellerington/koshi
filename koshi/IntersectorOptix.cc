#include <koshi/IntersectorOptix.h>

#include <optix_stubs.h>
#include <cuda_runtime_api.h>
#include <array>

#include <koshi/Scene.h>
#include <koshi/OptixHelpers.h>
#include <koshi/GeometryMesh.h>

KOSHI_OPEN_NAMESPACE

IntersectorOptix::IntersectorOptix(Scene * scene, OptixDeviceContext& context)
{
    auto geometries = scene->getGeometries();
    auto it = geometries.begin();
    if(it == geometries.end())
        return;

    // TODO: Dynamic cast this to check it's type?
    GeometryMesh * geometry = dynamic_cast<GeometryMesh*>(it->second);
    if(!geometry) return;
    GeometryMeshAttribute * vertices = geometry->getAttribute("vertices");
    if(!vertices) return;

    // accel handling
    {
        // Use default options for simplicity.  In a real use case we would want to
        // enable compaction, etc
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
        accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

        // // Triangle build input: simple list of three vertices
        // const std::array<Vec3f, 4> vertices =
        // { {
        //     Vec3f( -1.f, -1.f, -1.0f ),
        //     Vec3f(  1.f, -1.f, -1.0f ),
        //     Vec3f( -1.f,  1.f, -1.0f ),
        //     Vec3f(  1.f,  1.f, -1.0f )
        // } };
        // const size_t vertices_size = sizeof( Vec3f )*vertices.size();
        // CUdeviceptr d_vertices=0;
        // CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_vertices ), vertices_size ) );
        // CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_vertices ), vertices.data(), vertices_size, cudaMemcpyHostToDevice ) );

        // const std::array<Vec3u, 2> indices =
        // { {
        //     Vec3u( 0, 1, 2 ),
        //     Vec3u( 1, 2, 3 )
        // } };
        // const size_t indices_size = sizeof( Vec3u )*indices.size();
        // CUdeviceptr d_indices = 0;
        // CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_indices ), indices_size ) );
        // CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_indices ), indices.data(), indices_size, cudaMemcpyHostToDevice ) );

        // // Our build input is a simple list of non-indexed triangle vertices
        // const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
        // OptixBuildInput triangle_input = {};
        // triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        // triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
        // triangle_input.triangleArray.numVertices   = static_cast<uint32_t>( vertices.size() );
        // triangle_input.triangleArray.vertexBuffers = &d_vertices;

        // triangle_input.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        // triangle_input.triangleArray.indexStrideInBytes  = sizeof(Vec3u);
        // triangle_input.triangleArray.numIndexTriplets    = static_cast<uint32_t>( indices.size() );
        // triangle_input.triangleArray.indexBuffer         = d_indices;


        // Triangle build input: simple list of three vertices
        const size_t vertices_size = sizeof(float)*vertices->data_stride*vertices->data_size;
        CUdeviceptr d_vertices=0;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_vertices ), vertices_size ) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_vertices ), vertices->data, vertices_size, cudaMemcpyHostToDevice ) );

        const size_t indices_size = vertices->indices_size*3*sizeof(uint32_t);
        CUdeviceptr d_indices = 0;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_indices ), indices_size ) );
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_indices ), vertices->indices, indices_size, cudaMemcpyHostToDevice ) );

        // Our build input is a simple list of non-indexed triangle vertices
        const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
        OptixBuildInput triangle_input = {};
        triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.numVertices   = static_cast<uint32_t>( vertices->data_size );
        triangle_input.triangleArray.vertexBuffers = &d_vertices;

        triangle_input.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangle_input.triangleArray.indexStrideInBytes  = 3*sizeof(uint32_t);
        triangle_input.triangleArray.numIndexTriplets    = static_cast<uint32_t>( vertices->indices_size );
        triangle_input.triangleArray.indexBuffer         = d_indices;


        triangle_input.triangleArray.flags         = triangle_input_flags;
        triangle_input.triangleArray.numSbtRecords = 1;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage(
                    context,
                    &accel_options,
                    &triangle_input,
                    1, // Number of build inputs
                    &gas_buffer_sizes
                    ) );
        CUdeviceptr d_temp_buffer_gas;
        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &d_temp_buffer_gas ),
                    gas_buffer_sizes.tempSizeInBytes
                    ) );
        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &d_gas_output_buffer ),
                    gas_buffer_sizes.outputSizeInBytes
                    ) );

        OPTIX_CHECK( optixAccelBuild(
                    context,
                    0,                  // CUDA stream
                    &accel_options,
                    &triangle_input,
                    1,                  // num build inputs
                    d_temp_buffer_gas,
                    gas_buffer_sizes.tempSizeInBytes,
                    d_gas_output_buffer,
                    gas_buffer_sizes.outputSizeInBytes,
                    &gas_handle,
                    nullptr,            // emitted property list
                    0                   // num emitted properties
                    ) );

        CUDA_SYNC_CHECK();

        // We can now free the scratch space buffer used during build and the vertex
        // inputs, since they are not needed by our trivial shading method
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );
        CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_vertices        ) ) );

    }
}

KOSHI_CLOSE_NAMESPACE
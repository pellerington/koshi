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
    std::cout << "Start Intersector..." << std::endl;

    const std::unordered_map<std::string, Geometry*>& geometries = scene->getGeometries();

    instances.resize(geometries.size());
    uint i = 0;

    for(auto it = geometries.begin(); it != geometries.end(); ++it)
    {
        GeometryMesh * geometry = dynamic_cast<GeometryMesh*>(it->second);
        if(!geometry) continue;
        GeometryMeshAttribute * vertices = geometry->getAttribute("vertices");
        if(!vertices || vertices->format != Format::FLOAT32) continue;

        if(traversables.find(geometry) == traversables.end())
        {
            // Use default options for simplicity.
            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
            accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

            // Set the triangle array.
            const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
            OptixBuildInput triangle_input = {};
            triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangle_input.triangleArray.numVertices   = static_cast<uint32_t>(vertices->data_size);
            triangle_input.triangleArray.vertexBuffers = (CUdeviceptr*)&vertices->d_data;
            triangle_input.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangle_input.triangleArray.indexStrideInBytes  = sizeof(uint32_t)*vertices->indices_stride;
            triangle_input.triangleArray.numIndexTriplets    = static_cast<uint32_t>(vertices->indices_size);
            triangle_input.triangleArray.indexBuffer         = (CUdeviceptr)vertices->d_indices;;
            triangle_input.triangleArray.flags         = triangle_input_flags;
            triangle_input.triangleArray.numSbtRecords = 1;

            // Allocate the gas buffers.
            OptixAccelBufferSizes gas_buffer_sizes;
            OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &triangle_input, 1, &gas_buffer_sizes));
            CUdeviceptr temp_gas_buffer;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&temp_gas_buffer), gas_buffer_sizes.tempSizeInBytes));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&traversables[geometry].gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));

            // Perform accel build.
            OPTIX_CHECK(optixAccelBuild(context, 0, &accel_options, &triangle_input, 1, temp_gas_buffer, gas_buffer_sizes.tempSizeInBytes, 
                traversables[geometry].gas_output_buffer, gas_buffer_sizes.outputSizeInBytes, &traversables[geometry].traversable_handle, nullptr, 0));
            CUDA_SYNC_CHECK();
            
            // Free the temp gas buffer.
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(temp_gas_buffer)));
        }

        geometry->get_obj_to_world().copy(instances[i].transform);
        instances[i].instanceId = i;
        instances[i].visibilityMask = 255;
        instances[i].sbtOffset = 0;
        instances[i].flags = OPTIX_INSTANCE_FLAG_NONE;
        instances[i].traversableHandle = traversables[geometry].traversable_handle;

        i++;
    }

    // std::cout << "Made Instaces: " << i << std::endl;

    // Use default options for simplicity.
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

    // Setup instance input
    OptixBuildInput instance_input = {};
    instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&instance_input.instanceArray.instances), instances.size()*sizeof(OptixInstance)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(instance_input.instanceArray.instances), instances.data(), instances.size()*sizeof(OptixInstance), cudaMemcpyHostToDevice));
    instance_input.instanceArray.numInstances = instances.size();

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &instance_input, 1, &gas_buffer_sizes));
    CUdeviceptr temp_gas_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&temp_gas_buffer), gas_buffer_sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(context, 0, &accel_options, &instance_input, 1, temp_gas_buffer, gas_buffer_sizes.tempSizeInBytes, 
        gas_output_buffer, gas_buffer_sizes.outputSizeInBytes, &traversable_handle, nullptr, 0));

    CUDA_SYNC_CHECK();

    // Free the temp gas buffer.
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(temp_gas_buffer)));
}

KOSHI_CLOSE_NAMESPACE
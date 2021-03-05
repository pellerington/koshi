#pragma once

#include <optix_types.h>
#include <vector>
#include <unordered_map>

#include <koshi/Koshi.h>
#include <koshi/IntersectList.h>
#include <koshi/Ray.h>

KOSHI_OPEN_NAMESPACE

class Geometry;
class Scene;

// TODO: Make this extend base intersector class?

class IntersectorOptix
{
public:
    IntersectorOptix(Scene * scene, OptixDeviceContext& context);
    ~IntersectorOptix();

#ifdef CUDA_COMPILE
    DEVICE_FUNCTION IntersectList intersect(const Ray& ray)
    {
        float3 origin = make_float3(ray.origin.x, ray.origin.y, ray.origin.z);
        float3 direction = make_float3(ray.direction.x, ray.direction.y, ray.direction.z);

        IntersectList intersects;
        intersects.setRay(ray);

        // Payloads variables
        uint32_t p0, p1;
        const uint64_t ptr = reinterpret_cast<uint64_t>(&intersects);
        p0 = ptr >> 32; p1 = ptr & 0x00000000ffffffff;

        optixTrace(traversable_handle, 
            origin, direction,
            ray.tmin, ray.tmax, 0.0f /*time*/,
            OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
            0,      // SBT offset   -- See SBT discussion
            1,      // SBT stride   -- See SBT discussion
            0,      // missSBTIndex -- See SBT discussion
            p0, p1  // payload
        );

        return intersects;
    }
#endif

private:

    // TODO: We need device versions of these so we can intersect them.
    struct Traversable
    {
        OptixTraversableHandle traversable_handle;
        CUdeviceptr gas_output_buffer;
    };
    std::unordered_map<Geometry*, Traversable> traversables;
    std::vector<OptixInstance> instances;

    OptixTraversableHandle traversable_handle;
    CUdeviceptr            gas_output_buffer;

    // Reusable quad data.
    float * d_quad_vertices;
    uint32_t * d_quad_indices;
};

KOSHI_CLOSE_NAMESPACE
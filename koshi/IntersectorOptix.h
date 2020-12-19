#pragma once

#include <optix_types.h>
#include <vector>

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

#ifdef __CUDACC__
    DEVICE_FUNCTION IntersectList intersect(const Ray& ray)
    {
        float3 origin = make_float3(ray.origin.x, ray.origin.y, ray.origin.z);
        float3 direction = make_float3(ray.direction.x, ray.direction.y, ray.direction.z);

        IntersectList intersects;

        uint32_t p0, p1;
        const uint64_t ptr = reinterpret_cast<uint64_t>(&intersects);
        p0 = ptr >> 32; p1 = ptr & 0x00000000ffffffff;

        optixTrace(
            gas_handle,
            origin,
            direction,
            0.0f,                // Min intersection distance
            1e16f,               // Max intersection distance
            0.0f,                // rayTime -- used for motion blur
            OptixVisibilityMask( 255 ), // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset   -- See SBT discussion
            1,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            p0, p1
        );

        return intersects;
    }
#endif

private:
    std::vector<Geometry*> geometry;

    OptixTraversableHandle gas_handle;
    CUdeviceptr            d_gas_output_buffer;
};

KOSHI_CLOSE_NAMESPACE
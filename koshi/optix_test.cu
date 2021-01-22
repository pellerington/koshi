#include <optix.h>
#include <optix_device.h>

#include <koshi/RenderOptix.h>
#include <koshi/IntersectList.h>

using namespace Koshi;

extern "C" 
{
__constant__ Koshi::Resources resources;
}

DEVICE_FUNCTION IntersectList * unpackIntersects()
{ 
    const uint32_t ptr0 = optixGetPayload_0();
    const uint32_t ptr1 = optixGetPayload_1();
    const uint64_t ptr = static_cast<uint64_t>(ptr0) << 32 | ptr1;
    return reinterpret_cast<IntersectList*>(ptr); 
}

extern "C" __global__ void __raygen__rg() 
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const Ray ray = resources.camera->sample(idx.x, idx.y);
    IntersectList intersects = resources.intersector->intersect(ray);

    if(intersects.size() > 0)
    {
        Intersect& intersect = intersects[0];
        resources.aovs[0].write(Vec2u(idx.x, idx.y), Vec4f(intersect.uvw.u, intersect.uvw.v, intersect.uvw.w, 1.f));
    }
}

extern "C" __global__ void __miss__ms() 
{
}

extern "C" __global__ void __closesthit__ch() 
{
    IntersectList * intersects = unpackIntersects();
    const Ray& ray = intersects->getRay();

    Intersect& intersect = intersects->push();
    
    // Geometry = something...
    intersect.prim = optixGetPrimitiveIndex();

    intersect.t = optixGetRayTmax();

    // World to Object ?
    // Material * material / Integrator * integrator;
    
    // TODO: Should we store position or have it be calculated from t0.
    intersect.position = ray.origin + intersect.t * ray.direction;
    
    // intersect.normal = ;

    float2 uvs = optixGetTriangleBarycentrics();
    intersect.uvw = Vec3f(uvs.x, uvs.y, 0.f);
}
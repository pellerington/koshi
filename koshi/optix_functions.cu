#include <optix.h>
#include <optix_device.h>

#include <cfloat>

#include <koshi/RenderOptix.h>
#include <koshi/IntersectList.h>

#include <koshi/GeometryMesh.h>

#include <koshi/material/Material.h>

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

    const Ray ray = resources.camera->sample(idx.x, idx.y, Vec2f(resources.random->rand(), resources.random->rand()));

    const IntersectList intersects = resources.intersector->intersect(ray);

    if(intersects.size() > 0)
    {
        const Intersect& intersect = intersects[0];

        Aov * depth_aov = resources.getAov("depth");
        if(depth_aov)
            depth_aov->write(Vec2u(idx.x, idx.y), Vec4f(intersect.t, 0.f, 0.f, 1.f));

        Aov * normal_aov = resources.getAov("normal");
        if(normal_aov)
            normal_aov->write(Vec2u(idx.x, idx.y), Vec4f(intersect.normal, 1.f));

        if(!intersect.facing)
        {
            Aov * color_aov = resources.getAov("color");
            if(color_aov)
                color_aov->write(Vec2u(idx.x, idx.y), Vec4f(0.f, 0.f, 0.f, 1.f));
            return;
        }
        
        LobeArray lobes;
        generate_material(lobes, intersect);

        const Lambert * lambert = (const Lambert *)lobes[0];

        // Vec3f light_pos = ray.origin;//Vec3f(-30, -80, 200.f);//

        Vec3f color = 0.f;
        for(uint i = 0; i < 1; i++)
        {
            Sample sample;
            const Vec2f rnd(resources.random->rand(), resources.random->rand());
            lambert->sample(sample, rnd, intersect, ray.direction);

            Ray shadow_ray;
            shadow_ray.origin = intersect.position + intersect.normal * ((lobes[0]->getSide() == Lobe::FRONT) ? 0.0001f : -0.0001f); // TODO: Replace this with ray_bias;
            shadow_ray.direction = sample.wo;
            shadow_ray.tmin = 0.f;
            shadow_ray.tmax = FLT_MAX;
            const IntersectList shadow_intersects = resources.intersector->intersect(shadow_ray);

            if(shadow_intersects.empty())
            {
                color += sample.value / sample.pdf;
            }
        }
        color /= 1.f;

        Aov * color_aov = resources.getAov("color");
        if(color_aov)
            color_aov->write(Vec2u(idx.x, idx.y), Vec4f(color, 1.f));
    }
}

extern "C" __global__ void __miss__ms() 
{
    // Add distant light intersects.
}

extern "C" __global__ void __closesthit__ch() 
{
    IntersectList * intersects = unpackIntersects();
    const Ray& ray = intersects-> getRay();
    Intersect& intersect = intersects->push();
    
    intersect.prim = optixGetPrimitiveIndex();

    float m[12];
    optixGetObjectToWorldTransformMatrix(m);	
    intersect.obj_to_world = Transform::fromData(m);

    intersect.t = intersect.t_max = optixGetRayTmax();

    intersect.position = ray.origin + intersect.t * ray.direction;

    float2 uvs = optixGetTriangleBarycentrics();
    intersect.uvw = intersect.uvw_max = Vec3f(uvs.x, uvs.y, 0.f);

    HitGroupData * sbt_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    intersect.geometry = resources.scene->d_geometries[sbt_data->geometry_id];

    // TODO: Move this into it's own function.
    GeometryMesh * geometry_mesh = (GeometryMesh *)intersect.geometry;
    GeometryMeshAttribute * vertices = geometry_mesh->getVerticesAttribute();
    const uint p0 = vertices->d_indices[intersect.prim*vertices->indices_stride+0]*vertices->data_stride;
    const uint p1 = vertices->d_indices[intersect.prim*vertices->indices_stride+1]*vertices->data_stride;
    const uint p2 = vertices->d_indices[intersect.prim*vertices->indices_stride+2]*vertices->data_stride;
    const Vec3f v0 = Vec3f(((float*)vertices->d_data)[p0+0], ((float*)vertices->d_data)[p0+1], ((float*)vertices->d_data)[p0+2]);
    const Vec3f v1 = Vec3f(((float*)vertices->d_data)[p1+0], ((float*)vertices->d_data)[p1+1], ((float*)vertices->d_data)[p1+2]);
    const Vec3f v2 = Vec3f(((float*)vertices->d_data)[p2+0], ((float*)vertices->d_data)[p2+1], ((float*)vertices->d_data)[p2+2]);
    intersect.normal = intersect.obj_to_world.multiply<false>(Vec3f::cross(v1 - v0, v2 - v0).normalize());

    intersect.facing = optixIsFrontFaceHit();
}
#include <optix.h>
#include <optix_device.h>

#include <cfloat>

#include <koshi/RenderOptix.h>
#include <koshi/IntersectList.h>

#include <koshi/geometry/GeometryMesh.h>

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

        if(intersect.geometry->getType() != Geometry::MESH)
            return;

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
        generate_material(lobes, intersect, ray);

        if(lobes.empty())
            return;

        // const Lambert * lambert = (const Lambert *)lobes[0];

        Vec3f color = 0.f;
        const float num_samples = 2.f;
        for(uint i = 0; i < num_samples; i++)
        {
            const float lobe_prob = 1.f / lobes.size();
            const uint lobe_index = resources.random->rand() * lobes.size();
            const Lobe * lobe = lobes[lobe_index];

            Sample sample;
            const Vec2f rnd(resources.random->rand(), resources.random->rand());

            switch(lobe->getType())
            {
                case Lobe::LAMBERT: {
                    ((const Lambert *)lobe)->sample(sample, rnd, intersect, ray.direction);
                    break;
                }
                case Lobe::BACK_LAMBERT: {
                    ((const BackLambert *)lobe)->sample(sample, rnd, intersect, ray.direction);
                    break;
                }
                case Lobe::REFLECT: {
                    ((const Reflect *)lobe)->sample(sample, rnd, intersect, ray.direction);
                    break;
                }
            }
            sample.pdf *= lobe_prob;
    
            for(uint j = 0; j < lobes.size(); j++)
            {
                if(j == lobe_index)
                    continue;

                Sample eval_sample;
                bool evalated = false;
                switch(lobe->getType())
                {
                    case Lobe::LAMBERT: {
                        evalated = ((const Lambert *)lobe)->evaluate(eval_sample, intersect, ray.direction);
                        break;
                    }
                    case Lobe::BACK_LAMBERT: {
                        evalated = ((const BackLambert *)lobe)->evaluate(eval_sample, intersect, ray.direction);
                        break;
                    }
                    case Lobe::REFLECT: {
                        evalated = ((const Reflect *)lobe)->evaluate(eval_sample, intersect, ray.direction);
                        break;
                    }
                }
                
                if(evalated)
                {
                    sample.value += eval_sample.value;
                    sample.pdf += eval_sample.pdf * lobe_prob;
                }
            }

            const bool front = (sample.wo.dot(intersect.normal) > 0.f);
            if(lobes[0]->getSide() == Lobe::FRONT && !front || lobe->getSide() == Lobe::BACK && front)
                continue;

            Ray shadow_ray;
            shadow_ray.origin = intersect.position + intersect.normal * (front ? 0.0001f : -0.0001f); // TODO: Replace this with ray_bias;
            shadow_ray.direction = sample.wo;
            shadow_ray.tmin = 0.f;
            shadow_ray.tmax = FLT_MAX;
            const IntersectList shadow_intersects = resources.intersector->intersect(shadow_ray);

            if(!shadow_intersects.empty())
            {
                const Intersect& shadow_intersect = shadow_intersects[0];
                if(shadow_intersect.geometry->getType() == Geometry::ENVIRONMENT)
                {
                    GeometryEnvironment * env = (GeometryEnvironment *)shadow_intersect.geometry;
                    float4 out = tex2D<float4>(env->cuda_tex, shadow_intersect.uvw.u, shadow_intersect.uvw.v);
                    color += Vec3f(out.x, out.y, out.z) * sample.value / sample.pdf;

                    // if(lobe->getType() == Lobe::REFLECT)
                    //     printf("%f\n", (sample.pdf));
    
                }
            }
        }
        color /= num_samples;

        Aov * color_aov = resources.getAov("color");
        if(color_aov)
            color_aov->write(Vec2u(idx.x, idx.y), Vec4f(color, 1.f));
    }
}

extern "C" __global__ void __miss__ms() 
{
    IntersectList * intersects = unpackIntersects();
    const Ray& ray = intersects-> getRay();

    for(uint i = 0; i < resources.scene->num_distant_geometries; i++)
    {
        Intersect& intersect = intersects->push();
        
        intersect.prim = 0;
    
        HitGroupData * sbt_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
        intersect.geometry = resources.scene->d_geometries[resources.scene->d_distant_geometries[i]];
        intersect.obj_to_world = intersect.geometry->get_obj_to_world();
    
        intersect.t = intersect.t_max = FLT_MAX; // TODO: We should specify an infinite value
    
        intersect.position = Vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
    
        float theta = atanf((ray.direction.z + epsilon) / (ray.direction.x + epsilon));
        theta += ((ray.direction.z < 0.f) ? pi : 0.f) + ((ray.direction.z * ray.direction.x < 0.f) ? pi : 0.f);
        intersect.uvw = intersect.uvw_max = Vec3f(theta * inv_two_pi, acosf(ray.direction.y) * inv_pi, 0.f);
        
        intersect.normal = -ray.direction;
    
        intersect.facing = true;
    }
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
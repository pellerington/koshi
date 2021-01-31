#include <optix.h>
#include <optix_device.h>

#include <cfloat>

#include <koshi/RenderOptix.h>
#include <koshi/IntersectList.h>

#include <koshi/GeometryMesh.h>

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

    const IntersectList intersects = resources.intersector->intersect(ray);

    if(intersects.size() > 0)
    {
        const Intersect& intersect = intersects[0];

        if(!intersect.facing)
        {
            resources.aovs[0].write(Vec2u(idx.x, idx.y), Vec4f(0.f, 0.f, 0.f, 1.f));
            return;
        }

        Vec3f light_pos = ray.origin;//Vec3f(-30, -80, 200.f);//

        Ray shadow_ray;
        shadow_ray.origin = intersect.position + intersect.normal * 0.0001f;
        shadow_ray.direction = (light_pos - shadow_ray.origin).normalize();
        shadow_ray.tmin = 0.f;
        shadow_ray.tmax = (light_pos - shadow_ray.origin).length() - 0.0001f;

        const IntersectList shadow_intersects = resources.intersector->intersect(shadow_ray);

        if(shadow_intersects.size() > 0)
        {
            resources.aovs[0].write(Vec2u(idx.x, idx.y), Vec4f(0.f, 0.f, 0.f, 1.f));
        }
        else
        {
            GeometryMeshAttribute * normals = ((GeometryMesh *)intersect.geometry)->getAttribute("normals");
            GeometryMeshAttribute * display_color = ((GeometryMesh *)intersect.geometry)->getAttribute("displayColor");
            Vec3f color(1.f);
            color *= (normals) ? max(intersect.obj_to_world.multiply<false>(normals->evaluate(intersect)).dot(shadow_ray.direction), 0.f) : intersect.normal.dot(shadow_ray.direction);
            color *= (display_color) ? (Vec3f)display_color->evaluate(intersect) : 1.f;
            resources.aovs[0].write(Vec2u(idx.x, idx.y), Vec4f(color, 1.f));
        }
    }
}

extern "C" __global__ void __miss__ms() 
{
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
    GeometryMeshAttribute * vertices = geometry_mesh->getAttribute("vertices");
    const uint p0 = vertices->d_indices[intersect.prim*vertices->indices_stride+0]*vertices->data_stride;
    const uint p1 = vertices->d_indices[intersect.prim*vertices->indices_stride+1]*vertices->data_stride;
    const uint p2 = vertices->d_indices[intersect.prim*vertices->indices_stride+2]*vertices->data_stride;
    const Vec3f v0 = Vec3f(((float*)vertices->d_data)[p0+0], ((float*)vertices->d_data)[p0+1], ((float*)vertices->d_data)[p0+2]);
    const Vec3f v1 = Vec3f(((float*)vertices->d_data)[p1+0], ((float*)vertices->d_data)[p1+1], ((float*)vertices->d_data)[p1+2]);
    const Vec3f v2 = Vec3f(((float*)vertices->d_data)[p2+0], ((float*)vertices->d_data)[p2+1], ((float*)vertices->d_data)[p2+2]);
    intersect.normal = intersect.obj_to_world.multiply<false>(Vec3f::cross(v1 - v0, v2 - v0).normalize());

    intersect.facing = optixIsFrontFaceHit();
}
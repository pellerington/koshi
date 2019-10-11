#include "ObjectMesh.h"

ObjectMesh::ObjectMesh(uint _vertices_size, uint _triangles_size, uint _normals_size, VERT_DATA * _vertices, TRI_DATA * _tri_vindex, NORM_DATA * _normals, TRI_DATA * _tri_nindex,
                       const Transform3f &obj_to_world, std::shared_ptr<Material> material, std::shared_ptr<Volume> volume)
: Object(material, obj_to_world, volume),
  vertices_size(_vertices_size), triangles_size(_triangles_size), normals_size(_normals_size),
  vertices(_vertices), tri_vindex(_tri_vindex), normals(_normals), tri_nindex(_tri_nindex)
{
    //Transform our vertices and update bbox;
    bbox = Box3f();
    const bool transform = !obj_to_world.is_identity();
    for(uint i = 0; i < vertices_size; i++)
    {
        Vec3f v(vertices[i].x, vertices[i].y, vertices[i].z);
        if(transform) v = obj_to_world * v;
        bbox.extend(v);
        vertices[i].x = v.x; vertices[i].y = v.y; vertices[i].z = v.z;
    }
    for(uint i = 0; i < normals_size && transform; i++)
    {
        Vec3f n(normals[i].x, normals[i].y, normals[i].z);
        if(transform) n = obj_to_world.multiply(n, false);
        normals[i].x = n.x; normals[i].y = n.y; normals[i].z = n.z;
    }

    geom = rtcNewGeometry(Embree::rtc_device, RTC_GEOMETRY_TYPE_TRIANGLE);

    rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, vertices, 0, sizeof(VERT_DATA), vertices_size);
    rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, tri_vindex, 0, sizeof(TRI_DATA), triangles_size);
}

Surface ObjectMesh::process_intersection(const RTCRayHit &rtcRayHit, const Ray &ray)
{
    const Vec3f normal = (normals_size > 0)
    ? (Vec3f(normals[tri_nindex[rtcRayHit.hit.primID].v0].x, normals[tri_nindex[rtcRayHit.hit.primID].v0].y, normals[tri_nindex[rtcRayHit.hit.primID].v0].z) * (1.f - (rtcRayHit.hit.u + rtcRayHit.hit.v))
     + Vec3f(normals[tri_nindex[rtcRayHit.hit.primID].v1].x, normals[tri_nindex[rtcRayHit.hit.primID].v1].y, normals[tri_nindex[rtcRayHit.hit.primID].v1].z) * rtcRayHit.hit.u
     + Vec3f(normals[tri_nindex[rtcRayHit.hit.primID].v2].x, normals[tri_nindex[rtcRayHit.hit.primID].v2].y, normals[tri_nindex[rtcRayHit.hit.primID].v2].z) * rtcRayHit.hit.v).normalized()
    : Vec3f(rtcRayHit.hit.Ng_x, rtcRayHit.hit.Ng_y, rtcRayHit.hit.Ng_z).normalized();

    return Surface(
        ray.get_position(ray.t),
        normal,//Vec3f(rtcRayHit.hit.Ng_x, rtcRayHit.hit.Ng_y, rtcRayHit.hit.Ng_z).normalized(),
        // Smooth normals here (use ray.hit.primID to get triangle)
        // surface.normal = (smooth_normals) ? (1.f - u - v) * *normals[0] + u * *normals[1] + v * *normals[2] : normal;
        ray.dir,
        rtcRayHit.hit.u,
        rtcRayHit.hit.v
    );
}

ObjectMesh::~ObjectMesh()
{
    delete[] vertices;
    delete[] tri_vindex;
    delete[] normals;
    delete[] tri_nindex;
}

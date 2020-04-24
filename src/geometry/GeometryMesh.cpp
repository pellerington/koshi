#include <geometry/GeometryMesh.h>

GeometryMesh::GeometryMesh(const Transform3f &obj_to_world,
                           uint _vertices_size, uint _triangles_size, 
                           uint _normals_size, uint _uvs_size,
                           VERT_DATA * _vertices, TRI_DATA * _tri_vert_index,
                           NORM_DATA * _normals, TRI_DATA * _tri_norm_index,
                           UV_DATA * _uvs, TRI_DATA * _tri_uvs_index)
: Geometry(obj_to_world),
  vertices_size(_vertices_size), triangles_size(_triangles_size), normals_size(_normals_size), uvs_size(_uvs_size),
  vertices(_vertices), tri_vert_index(_tri_vert_index), normals(_normals), tri_norm_index(_tri_norm_index), uvs(_uvs), tri_uvs_index(_tri_uvs_index)
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
}

// void ObjectMesh::process_intersection(Surface &surface, const RTCRayHit &rtcRayHit, const Ray &ray)
// {
//     const Vec3f normal = (normals_size > 0)
//     ? (Vec3f(normals[tri_norm_index[rtcRayHit.hit.primID].v0].x, normals[tri_norm_index[rtcRayHit.hit.primID].v0].y, normals[tri_norm_index[rtcRayHit.hit.primID].v0].z) * (1.f - (rtcRayHit.hit.u + rtcRayHit.hit.v))
//      + Vec3f(normals[tri_norm_index[rtcRayHit.hit.primID].v1].x, normals[tri_norm_index[rtcRayHit.hit.primID].v1].y, normals[tri_norm_index[rtcRayHit.hit.primID].v1].z) * rtcRayHit.hit.u
//      + Vec3f(normals[tri_norm_index[rtcRayHit.hit.primID].v2].x, normals[tri_norm_index[rtcRayHit.hit.primID].v2].y, normals[tri_norm_index[rtcRayHit.hit.primID].v2].z) * rtcRayHit.hit.v).normalized()
//     : Vec3f(rtcRayHit.hit.Ng_x, rtcRayHit.hit.Ng_y, rtcRayHit.hit.Ng_z).normalized();

//     float u = rtcRayHit.hit.u, v = rtcRayHit.hit.v;
//     if(uvs_size > 0)
//     {
//         const float v1_weight = rtcRayHit.hit.u;
//         const float v2_weight = rtcRayHit.hit.v;
//         const float v0_weight = 1.f - rtcRayHit.hit.u - rtcRayHit.hit.v;
//         u = v0_weight * uvs[tri_uvs_index[rtcRayHit.hit.primID].v0].u + v1_weight * uvs[tri_uvs_index[rtcRayHit.hit.primID].v1].u + v2_weight * uvs[tri_uvs_index[rtcRayHit.hit.primID].v2].u;
//         v = v0_weight * uvs[tri_uvs_index[rtcRayHit.hit.primID].v0].v + v1_weight * uvs[tri_uvs_index[rtcRayHit.hit.primID].v1].v + v2_weight * uvs[tri_uvs_index[rtcRayHit.hit.primID].v2].v;
//     }

//     surface.set_hit
//     (
//         ray.get_position(ray.t),
//         normal,
//         Vec3f(rtcRayHit.hit.Ng_x, rtcRayHit.hit.Ng_y, rtcRayHit.hit.Ng_z).normalized(),
//         u, v
//     );
// }

GeometryMesh::~GeometryMesh()
{
    delete[] vertices;
    delete[] tri_vert_index;
    delete[] normals;
    delete[] tri_norm_index;
    delete[] uvs;
    delete[] tri_uvs_index;
}

#pragma once

#include <geometry/Object.h>
#include <Math/Types.h>
#include <vector>

class ObjectMesh : public Object
{
public:
    ObjectMesh(uint _vertices_size, uint _triangles_size, uint _normals_size, uint _uvs_size,
               VERT_DATA * _vertices, TRI_DATA * _tri_vert_index,
               NORM_DATA * _normals, TRI_DATA * _tri_norm_index,
               UV_DATA * _uvs, TRI_DATA * _tri_uvs_index,
               const Transform3f &obj_to_world, std::shared_ptr<Material> material = nullptr, std::shared_ptr<Volume> volume = nullptr);

    Type get_type() { return Object::Mesh; }
    void process_intersection(Surface &surface, const RTCRayHit &rtcRayHit, const Ray &ray);

    ~ObjectMesh();

private:
    uint vertices_size;
    uint triangles_size;
    uint normals_size;
    uint uvs_size;

    VERT_DATA * vertices;
    TRI_DATA * tri_vert_index;

    NORM_DATA * normals;
    TRI_DATA * tri_norm_index;

    UV_DATA * uvs;
    TRI_DATA * tri_uvs_index;
};

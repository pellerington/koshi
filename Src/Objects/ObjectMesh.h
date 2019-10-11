#pragma once

#include "Object.h"
#include "../Math/Types.h"
#include <vector>

class ObjectMesh : public Object
{
public:
    ObjectMesh(uint _vertices_size, uint _triangles_size, uint _normals_size, VERT_DATA * _vertices, TRI_DATA * _tri_vindex, NORM_DATA * _normals, TRI_DATA * _tri_nindex,
               const Transform3f &obj_to_world, std::shared_ptr<Material> material = nullptr, std::shared_ptr<Volume> volume = nullptr);

    Type get_type() { return Object::Mesh; }
    Surface process_intersection(const RTCRayHit &rtcRayHit, const Ray &ray);

    ~ObjectMesh();

private:
    uint vertices_size;
    uint triangles_size;
    uint normals_size;

    VERT_DATA * vertices;
    TRI_DATA * tri_vindex;
    NORM_DATA * normals;
    TRI_DATA * tri_nindex;
};

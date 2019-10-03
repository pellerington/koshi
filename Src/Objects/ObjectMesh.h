#pragma once

#include "ObjectTriangle.h"

#include <vector>

class ObjectTriangle;

class ObjectMesh : public Object
{
public:
    struct TriangleData
    {
        uint v_index[3];
        uint n_index[3];
    };

    ObjectMesh(const std::vector<Vec3f> &vertices, const std::vector<TriangleData> &triangle_data, std::shared_ptr<Material> material = nullptr);
    ObjectMesh(const std::vector<Vec3f> &vertices, const std::vector<Vec3f> &normals, const std::vector<TriangleData> &triangle_data, std::shared_ptr<Material> material = nullptr);

    Type get_type() { return Object::Mesh; }
    Surface process_intersection(const RTCRayHit &rtcRayHit, const Ray &ray);

private:
    RTCVertex * vertices; // These need to be correctly deleted in ~ObjectMesh()
    RTCTriangle * triangles;
    std::vector<std::array<std::shared_ptr<Vec3f>, 3>> normals;

};

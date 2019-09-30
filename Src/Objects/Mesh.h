#pragma once

#include "Triangle.h"

#include <vector>

class Triangle;

class Mesh : public Object
{
public:
    struct TriangleData
    {
        uint v_index[3];
        uint n_index[3];
    };

    Mesh(const std::vector<Vec3f> &vertices, const std::vector<TriangleData> &triangle_data, std::shared_ptr<Material> material = nullptr);
    Mesh(const std::vector<Vec3f> &vertices, const std::vector<Vec3f> &normals, const std::vector<TriangleData> &triangle_data, std::shared_ptr<Material> material = nullptr);

    ObjectType get_type() { return ObjectType::Mesh; }
    Surface process_intersection(const RTCRayHit &rtcRayHit, const Ray &ray);

private:
    RTCVertex * vertices; // These need to be correctly deleted in ~Mesh()
    RTCTriangle * triangles;
    std::vector<std::array<std::shared_ptr<Vec3f>, 3>> normals;

};

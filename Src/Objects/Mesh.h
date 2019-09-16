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

    Mesh(std::vector<Vec3f> &vertices, std::vector<TriangleData> &triangle_data, std::shared_ptr<Material> material = nullptr);
    Mesh(std::vector<Vec3f> &vertices, std::vector<Vec3f> &normals, std::vector<TriangleData> &triangle_data, std::shared_ptr<Material> material = nullptr);

    ObjectType get_type() { return ObjectType::Mesh; }
    bool intersect(Ray &ray, Surface &surface);
    std::vector<std::shared_ptr<Object>> get_sub_objects();

#if EMBREE
    void process_intersection(RTCRayHit &rtcRayHit, Ray &ray, Surface &surface);
#endif

private:

#if EMBREE
    RTCVertex * vertices; // These need to be correctly deleted in ~Mesh()
    RTCTriangle * triangles;
    std::vector<std::array<std::shared_ptr<Vec3f>, 3>> normals;
#else
    std::vector<Vec3f> vertices;
    std::vector<std::shared_ptr<Triangle>> triangles;
    std::vector<Vec3f> normals;
#endif

};

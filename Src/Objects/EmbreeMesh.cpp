#include "Mesh.h"

#if EMBREE

Mesh::Mesh(std::vector<Vec3f> &_vertices, std::vector<TriangleData> &triangle_data, std::shared_ptr<Material> material)
: Object(material)
{
    mesh = rtcNewGeometry(Embree::rtc_device, RTC_GEOMETRY_TYPE_TRIANGLE);

    vertices = (RTCVertex*) rtcSetNewGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(RTCVertex), _vertices.size());
    for(uint i = 0; i < _vertices.size(); i++)
    {
        vertices[i].x = _vertices[i][0];
        vertices[i].y = _vertices[i][1];
        vertices[i].z = _vertices[i][2];
    }

    triangles = (RTCTriangle*) rtcSetNewGeometryBuffer(mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(RTCTriangle), triangle_data.size());
    for(uint i = 0; i < triangle_data.size(); i++)
    {
        triangles[i].v0 = triangle_data[i].v_index[0];
        triangles[i].v1 = triangle_data[i].v_index[1];
        triangles[i].v2 = triangle_data[i].v_index[2];
    }
}

Mesh::Mesh(std::vector<Vec3f> &_vertices, std::vector<Vec3f> &_normals, std::vector<TriangleData> &triangle_data, std::shared_ptr<Material> material)
: Object(material)
{
    mesh = rtcNewGeometry(Embree::rtc_device, RTC_GEOMETRY_TYPE_TRIANGLE);

    vertices = (RTCVertex*) rtcSetNewGeometryBuffer(mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(RTCVertex), _vertices.size());
    for(uint i = 0; i < _vertices.size(); i++)
    {
        vertices[i].x = _vertices[i][0];
        vertices[i].y = _vertices[i][1];
        vertices[i].z = _vertices[i][2];
    }

    triangles = (RTCTriangle*) rtcSetNewGeometryBuffer(mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(RTCTriangle), triangle_data.size());
    for(uint i = 0; i < triangle_data.size(); i++)
    {
        triangles[i].v0 = triangle_data[i].v_index[0];
        triangles[i].v1 = triangle_data[i].v_index[1];
        triangles[i].v2 = triangle_data[i].v_index[2];
    }

    // std::vector<std::shared_ptr<Vec3f>> normal_ptrs;
    // normal_ptrs.reserve(_normals.size());
    // std::transform(_normals.begin(), _normals.end(), normal_ptrs.begin(), [](Vec3f& v) { return std::make_shared<Vec3f>(v); });
    // normals.resize(triangle_data.size());
    // for(uint i = 0; i < normals.size(); i++)
    // {
    //     normals[i][0] = normal_ptrs[triangle_data[i].n_index[0]];
    //     normals[i][1] = normal_ptrs[triangle_data[i].n_index[1]];
    //     normals[i][2] = normal_ptrs[triangle_data[i].n_index[2]];
    // }
}

void Mesh::process_intersection(RTCRayHit &rtcRayHit, Ray &ray, Surface &surface)
{
    surface.position = ray.o + ray.t * ray.dir;
    surface.wi = ray.dir;
    surface.normal = Vec3f(rtcRayHit.hit.Ng_x, rtcRayHit.hit.Ng_y, rtcRayHit.hit.Ng_z);
    surface.normal.normalize();
    surface.enter = surface.normal.dot(ray.dir) < 0;
    surface.u = rtcRayHit.hit.u;
    surface.v = rtcRayHit.hit.v;
    //Smooth normals here (use ray.hit.primID to get triangle)
    // surface.normal = (smooth_normals) ? (1.f - u - v) * *normals[0] + u * *normals[1] + v * *normals[2] : normal;
    surface.object = this;
}

bool Mesh::intersect(Ray &ray, Surface &surface)
{
    return false; // Fill this in at some point;
}

std::vector<std::shared_ptr<Object>> Mesh::get_sub_objects()
{
    return std::vector<std::shared_ptr<Object>>(1, std::shared_ptr<Object>(this));
}

#endif

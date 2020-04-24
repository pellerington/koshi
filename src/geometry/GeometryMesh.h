#pragma once

#include <geometry/Geometry.h>
#include <Math/Types.h>
#include <vector>

class GeometryMesh : public Geometry
{
public:
    GeometryMesh(const Transform3f &obj_to_world, 
                 uint _vertices_size, uint _triangles_size, 
                 uint _normals_size, uint _uvs_size,
                 VERT_DATA * _vertices, TRI_DATA * _tri_vert_index,
                 NORM_DATA * _normals, TRI_DATA * _tri_norm_index,
                 UV_DATA * _uvs, TRI_DATA * _tri_uvs_index);


    // Replace these with actual attributes
    VERT_DATA * get_vertices() { return vertices; }
    uint get_vertices_size() { return vertices_size; }
    TRI_DATA * get_indicies() { return tri_vert_index; }
    uint get_triangles_size() { return triangles_size; }

    ~GeometryMesh();

private:
    // Make ALL attributes
    uint vertices_size;
    uint triangles_size;
    uint normals_size;
    uint uvs_size;
    // Make ALL attributes
    VERT_DATA * vertices;
    TRI_DATA * tri_vert_index;
    // Make ALL attributes
    NORM_DATA * normals;
    TRI_DATA * tri_norm_index;
    // Make ALL attributes
    UV_DATA * uvs;
    TRI_DATA * tri_uvs_index;
};

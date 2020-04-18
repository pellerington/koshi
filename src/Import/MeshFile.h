#pragma once

#include <geometry/GeometryMesh.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

class MeshFile
{
public:
    static std::shared_ptr<GeometryMesh> ImportOBJ(const std::string filename, const Transform3f &transform, std::shared_ptr<Material> material, std::shared_ptr<Volume> volume = nullptr)
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;

        std::string warn;
        std::string err;

        bool loaded_obj = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str());

        if (!err.empty())
            std::cerr << err << std::endl;
        if (!loaded_obj)
            return std::shared_ptr<GeometryMesh>();

        uint num_vertices = attrib.vertices.size() / 3;
        VERT_DATA * vertices = new VERT_DATA[num_vertices];
        for(size_t v = 0; v < num_vertices; v++)
        {
            vertices[v].x = attrib.vertices[v*3+0];
            vertices[v].y = attrib.vertices[v*3+1];
            vertices[v].z = attrib.vertices[v*3+2];
        }

        uint num_normals = attrib.normals.size() / 3;
        NORM_DATA * normals = nullptr;
        if(num_normals > 0)
        {
            normals = new NORM_DATA[num_normals];
            for(uint v = 0; v < num_normals; v++)
            {
                normals[v].x = attrib.normals[v*3+0];
                normals[v].y = attrib.normals[v*3+1];
                normals[v].z = attrib.normals[v*3+2];
            }
        }

        uint num_uvs = attrib.texcoords.size() / 2;
        UV_DATA * uvs = nullptr;
        if(num_uvs > 0)
        {
            uvs = new UV_DATA[num_uvs];
            for(uint v = 0; v < num_uvs; v++)
            {
                uvs[v].u = attrib.texcoords[v*2+0];
                uvs[v].v = attrib.texcoords[v*2+1];
            }
        }

        uint num_triangles = 0;
        for (size_t s = 0; s < shapes.size(); s++)
            num_triangles += shapes[s].mesh.num_face_vertices.size();

        TRI_DATA * tri_vert_index = new TRI_DATA[num_triangles];

        TRI_DATA * tri_norm_index = nullptr;
        if(num_normals) tri_norm_index = new TRI_DATA[num_triangles];

        TRI_DATA * tri_uvs_index = nullptr;
        if(num_uvs) tri_uvs_index = new TRI_DATA[num_triangles];

        uint offset = 0;
        for (size_t s = 0; s < shapes.size(); s++)
        {
            for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
            {
                // Allocate triangles, assuming triangulation.
                tinyobj::index_t idx = shapes[s].mesh.indices[(f * 3) + 0];
                tri_vert_index[f + offset].v0 = idx.vertex_index;
                if(num_normals) tri_norm_index[f + offset].v0 = idx.normal_index;
                if(num_uvs) tri_uvs_index[f + offset].v0 = idx.texcoord_index;

                idx = shapes[s].mesh.indices[(f * 3) + 1];
                tri_vert_index[f + offset].v1 = idx.vertex_index;
                if(num_normals) tri_norm_index[f + offset].v1 = idx.normal_index;
                if(num_uvs) tri_uvs_index[f + offset].v1 = idx.texcoord_index;

                idx = shapes[s].mesh.indices[(f * 3) + 2];
                tri_vert_index[f + offset].v2 = idx.vertex_index;
                if(num_normals) tri_norm_index[f + offset].v2 = idx.normal_index;
                if(num_uvs) tri_uvs_index[f + offset].v2 = idx.texcoord_index;
            }
            offset += shapes[s].mesh.num_face_vertices.size();
        }

        return std::shared_ptr<GeometryMesh>(new GeometryMesh(num_vertices, num_triangles, num_normals, num_uvs,
                                                              vertices, tri_vert_index,
                                                              normals, tri_norm_index,
                                                              uvs, tri_uvs_index,
                                                              transform, material));
    }
};

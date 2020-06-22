#pragma once

#include <geometry/GeometryMesh.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

class MeshFile
{
public:
    static std::vector<GeometryMeshAttribute*> ReadOBJ(const std::string& filename)
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn;
        std::string err;

        // Read File
        std::vector<GeometryMeshAttribute*> attributes;
        bool loaded = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str());
        if (!err.empty()) std::cerr << err << std::endl;
        if (!loaded || attrib.vertices.size() <= 0) return attributes;

        uint num_triangles = 0;
        for (size_t s = 0; s < shapes.size(); s++)
            num_triangles += shapes[s].mesh.num_face_vertices.size();

        // Load Vertices.
        GeometryMeshAttribute * vertices = new GeometryMeshAttribute;
        vertices->name = "vertices";
        vertices->array_size = attrib.vertices.size() / 3;
        vertices->array = new float4[vertices->array_size];
        float4 * vertices_array = (float4*)vertices->array;
        for(uint i = 0; i < vertices->array_size; i++)
        {
            vertices_array[i].v0 = attrib.vertices[i*3+0];
            vertices_array[i].v1 = attrib.vertices[i*3+1];
            vertices_array[i].v2 = attrib.vertices[i*3+2];
        }
        vertices->indices_size = num_triangles;
        vertices->indices = new uint3[num_triangles];
        attributes.push_back(vertices);

        GeometryMeshAttribute * normals = nullptr;
        if(attrib.normals.size() > 0)
        {
            normals = new GeometryMeshAttribute;
            normals->name = "normals";
            normals->array_size = attrib.normals.size() / 3;
            normals->array = new float3[normals->array_size];
            float3 * normals_array = (float3*)normals->array;
            for(uint i = 0; i < normals->array_size; i++)
            {
                normals_array[i].v0 = attrib.normals[i*3+0];
                normals_array[i].v1 = attrib.normals[i*3+1];
                normals_array[i].v2 = attrib.normals[i*3+2];
            }
            normals->indices_size = num_triangles;
            normals->indices = new uint3[num_triangles];
            attributes.push_back(normals);
        }

        GeometryMeshAttribute * uvs = nullptr;
        if(attrib.texcoords.size() > 0)
        {
            uvs = new GeometryMeshAttribute;
            uvs->name = "uvs";
            uvs->array_size = attrib.texcoords.size() / 2;
            uvs->array = new float2[uvs->array_size];
            float2 * uvs_array = (float2*)uvs->array;
            for(uint i = 0; i < uvs->array_size; i++)
            {
                uvs_array[i].v0 = attrib.texcoords[i*2+0];
                uvs_array[i].v1 = attrib.texcoords[i*2+1];
            }
            uvs->indices_size = num_triangles;
            uvs->indices = new uint3[num_triangles];
            attributes.push_back(uvs);
        }

        uint offset = 0;
        for (size_t s = 0; s < shapes.size(); s++)
        {
            for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
            {
                // Allocate triangles, assuming triangulation.
                tinyobj::index_t idx = shapes[s].mesh.indices[(f * 3) + 0];
                vertices->indices[f + offset].v0 = idx.vertex_index;
                if(normals) normals->indices[f + offset].v0 = idx.normal_index;
                if(uvs) uvs->indices[f + offset].v0 = idx.texcoord_index;

                idx = shapes[s].mesh.indices[(f * 3) + 1];
                vertices->indices[f + offset].v1 = idx.vertex_index;
                if(normals) normals->indices[f + offset].v1 = idx.normal_index;
                if(uvs) uvs->indices[f + offset].v1 = idx.texcoord_index;

                idx = shapes[s].mesh.indices[(f * 3) + 2];
                vertices->indices[f + offset].v2 = idx.vertex_index;
                if(normals) normals->indices[f + offset].v2 = idx.normal_index;
                if(uvs) uvs->indices[f + offset].v2 = idx.texcoord_index;
            }
            offset += shapes[s].mesh.num_face_vertices.size();
        }

        // TODO: DO WE NEED TO CLEANUP TINYOBJ ???

        return attributes;
    }
};

#pragma once

#include <koshi/geometry/GeometryMesh.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <koshi/dependency/tiny_obj_loader.h>

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
        vertices->array_item_size = 4;
        vertices->array_item_pad = 1; 
        vertices->array_item_count = attrib.vertices.size() / 3;
        vertices->array = new float[vertices->array_item_count * vertices->array_item_size];
        for(uint i = 0; i < vertices->array_item_count; i++)
        {
            vertices->array[i*vertices->array_item_size + 0] = attrib.vertices[i*3+0];
            vertices->array[i*vertices->array_item_size + 1] = attrib.vertices[i*3+1];
            vertices->array[i*vertices->array_item_size + 2] = attrib.vertices[i*3+2];
        }
        vertices->indices_item_size = 3;
        vertices->indices_item_pad = 0;
        vertices->indices_item_count = num_triangles;
        vertices->indices = new uint[vertices->indices_item_count * vertices->indices_item_size];
        attributes.push_back(vertices);

        // Load Normals.
        GeometryMeshAttribute * normals = nullptr;
        if(attrib.normals.size() > 0)
        {
            normals = new GeometryMeshAttribute;
            normals->name = "normals";
            normals->array_item_size = 3;
            normals->array_item_pad = 0;
            normals->array_item_count = attrib.normals.size() / 3;
            normals->array = new float[normals->array_item_count * normals->array_item_size];
            for(uint i = 0; i < normals->array_item_count; i++)
            {
                normals->array[i*normals->array_item_size + 0] = attrib.normals[i*3+0];
                normals->array[i*normals->array_item_size + 1] = attrib.normals[i*3+1];
                normals->array[i*normals->array_item_size + 2] = attrib.normals[i*3+2];
            }
            normals->indices_item_size = 3;
            normals->indices_item_pad = 0;
            normals->indices_item_count = num_triangles;
            normals->indices = new uint[normals->indices_item_count * normals->indices_item_size];
            attributes.push_back(normals);
        }

        // Load UVs.
        GeometryMeshAttribute * uvs = nullptr;
        if(attrib.texcoords.size() > 0)
        {
            uvs = new GeometryMeshAttribute;
            uvs->name = "uvs";
            uvs->array_item_size = 2;
            uvs->array_item_pad = 0;
            uvs->array_item_count = attrib.texcoords.size() / 2;
            uvs->array = new float[uvs->array_item_count * uvs->array_item_size];
            for(uint i = 0; i < uvs->array_item_count; i++)
            {
                uvs->array[i*uvs->array_item_size + 0] = attrib.texcoords[i*2+0];
                uvs->array[i*uvs->array_item_size + 1] = attrib.texcoords[i*2+1];
            }
            uvs->indices_item_size = 3;
            uvs->indices_item_pad = 0;
            uvs->indices_item_count = num_triangles;
            uvs->indices = new uint[uvs->indices_item_count * uvs->indices_item_size];
            attributes.push_back(uvs);
        }

        uint offset = 0;
        for (size_t s = 0; s < shapes.size(); s++)
        {
            for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
            {
                // Allocate triangles, assuming triangulation.
                tinyobj::index_t idx = shapes[s].mesh.indices[(f * 3) + 0];
                vertices->indices[(f + offset)*vertices->indices_item_size + 0] = idx.vertex_index;
                if(normals) normals->indices[(f + offset)*normals->indices_item_size + 0] = idx.normal_index;
                if(uvs) uvs->indices[(f + offset)*uvs->indices_item_size + 0] = idx.texcoord_index;

                idx = shapes[s].mesh.indices[(f * 3) + 1];
                vertices->indices[(f + offset)*vertices->indices_item_size + 1] = idx.vertex_index;
                if(normals) normals->indices[(f + offset)*normals->indices_item_size + 1] = idx.normal_index;
                if(uvs) uvs->indices[(f + offset)*uvs->indices_item_size + 1] = idx.texcoord_index;

                idx = shapes[s].mesh.indices[(f * 3) + 2];
                vertices->indices[(f + offset)*vertices->indices_item_size + 2] = idx.vertex_index;
                if(normals) normals->indices[(f + offset)*normals->indices_item_size + 2] = idx.normal_index;
                if(uvs) uvs->indices[(f + offset)*uvs->indices_item_size + 2] = idx.texcoord_index;
            }
            offset += shapes[s].mesh.num_face_vertices.size();
        }

        // TODO: DO WE NEED TO CLEANUP TINYOBJ ???

        return attributes;
    }
};

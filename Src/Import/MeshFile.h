#pragma once

#include "../Objects/Mesh.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "../Dependency/tiny_obj_loader.h"

class MeshFile
{
public:
    static std::shared_ptr<Mesh> ImportOBJ(std::string filename, std::shared_ptr<Material> material)
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
            return std::shared_ptr<Mesh>();

        std::vector<Vec3f> vertices;
        vertices.reserve(attrib.vertices.size() / 3.0);
        for(size_t v = 0; v < attrib.vertices.size(); v=v+3)
            vertices.emplace_back(Vec3f(attrib.vertices[v+0], attrib.vertices[v+1], attrib.vertices[v+2]));
        std::vector<Vec3f> normals;
        normals.reserve(attrib.normals.size() / 3.0);
        for(size_t v = 0; v < attrib.normals.size(); v=v+3)
            normals.emplace_back(Vec3f(attrib.normals[v+0], attrib.normals[v+1], attrib.normals[v+2]));

        std::vector<Mesh::TriangleData> triangle_data;
        for (size_t s = 0; s < shapes.size(); s++)
        {
            triangle_data.reserve(triangle_data.size() + shapes[s].mesh.num_face_vertices.size());
            for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
            {
                Mesh::TriangleData data;
                for (size_t v = 0; v < 3; v++)
                {
                      tinyobj::index_t idx = shapes[s].mesh.indices[(f * 3) + v]; //This +3 is assuming trianglation
                      data.v_index[v] = idx.vertex_index;
                      data.n_index[v] = idx.normal_index; // Todo: perform check that normal exists?
                      // tinyobj::real_t tx = attrib.texcoords[2*idx.texcoord_index+0];
                      // tinyobj::real_t ty = attrib.texcoords[2*idx.texcoord_index+1];
                }
                triangle_data.emplace_back(data);
            }
        }

        return std::shared_ptr<Mesh>(new Mesh(vertices, normals, triangle_data, material));
    }
};

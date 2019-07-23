#pragma once

#include "../Objects/Mesh.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "../Dependency/tiny_obj_loader.h"

class MeshFile
{
public:
    static std::shared_ptr<Mesh> ImportOBJ(std::string filename)
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

        std::vector<std::shared_ptr<Vec3f>> vert;
        vert.reserve(attrib.vertices.size() / 3.0);
        for(size_t v = 0; v < attrib.vertices.size(); v=v+3)
            vert.emplace_back(new Vec3f(attrib.vertices[v+0], attrib.vertices[v+1], attrib.vertices[v+2]));
        std::vector<std::shared_ptr<Vec3f>> normals;
        normals.reserve(attrib.normals.size() / 3.0);
        for(size_t v = 0; v < attrib.normals.size(); v=v+3)
            normals.emplace_back(new Vec3f(attrib.normals[v+0], attrib.normals[v+1], attrib.normals[v+2]));



        std::vector<std::shared_ptr<Triangle>> triangle;
        for (size_t s = 0; s < shapes.size(); s++)
        {
            triangle.reserve(triangle.size() + shapes[s].mesh.num_face_vertices.size());
            for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
            {
                uint vert_indices[3];
                uint normal_indices[3];
                for (size_t v = 0; v < 3; v++)
                {
                      tinyobj::index_t idx = shapes[s].mesh.indices[(f * 3) + v]; //This +3 is assuming trianglation
                      vert_indices[v] = idx.vertex_index;
                      normal_indices[v] = idx.normal_index; // Todo: perform check that normal exists

                      // tinyobj::real_t tx = attrib.texcoords[2*idx.texcoord_index+0];
                      // tinyobj::real_t ty = attrib.texcoords[2*idx.texcoord_index+1];
                }
                triangle.emplace_back(new Triangle(vert[vert_indices[0]], vert[vert_indices[1]], vert[vert_indices[2]],
                                                   normals[normal_indices[0]], normals[normal_indices[1]], normals[normal_indices[2]]));
            }
        }

        return std::shared_ptr<Mesh>(new Mesh(vert, normals, triangle));
    }
};

#pragma once

#include <geometry/Geometry.h>
#include <Math/Types.h>
#include <vector>

// TODO: Make this nicer and more extendable. Storing vertices as float4 but we don't want it to read array[4].
struct GeometryMeshAttribute 
{
    std::string name;
    void * array;

    // STORE THESE AS DATA TYPE __m128 for vertices and float[3], float[2] for others so we can use an array.

    uint array_size;

    uint3 * indices;
    uint indices_size;
};

class GeometryMesh : public Geometry
{
public:
    GeometryMesh(const Transform3f &obj_to_world, const std::string& filename);

    const GeometryMeshAttribute * get_mesh_attribute(const std::string& name)
    {
        auto attribute = attributes.find(name);
        return (attribute != attributes.end()) ? attribute->second : nullptr;
    }

    ~GeometryMesh();

private:
    std::unordered_map<std::string, GeometryMeshAttribute*> attributes;
};

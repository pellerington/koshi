#pragma once

#include <geometry/Geometry.h>
#include <Math/Types.h>
#include <vector>

struct GeometryMeshAttribute 
{
    std::string name;
    
    float * array;
    uint array_item_size;
    uint array_item_pad;
    uint array_item_count;

    uint * indices;
    uint indices_item_size;
    uint indices_item_pad;
    uint indices_item_count;
};

class GeometryMesh : public Geometry
{
public:
    GeometryMesh(const Transform3f &obj_to_world, const std::string& filename);

    const GeometryMeshAttribute * get_mesh_attribute(const std::string& name);

    bool eval_geometry_attribute(Vec3f& out, const std::string& attribute_name, const float& u, const float& v, const float& w, const uint& prim, Resources& resources);

    ~GeometryMesh();

private:
    std::unordered_map<std::string, GeometryMeshAttribute*> mesh_attributes;
};

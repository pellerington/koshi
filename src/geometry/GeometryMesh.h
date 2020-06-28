#pragma once

#include <geometry/Geometry.h>
#include <Math/Types.h>
#include <vector>

class GeometryMeshAttribute  : public GeometryAttribute
{
public:
    std::string name;
    
    float * array;
    uint array_item_size;
    uint array_item_pad;
    uint array_item_count;

    uint * indices;
    uint indices_item_size;
    uint indices_item_pad;
    uint indices_item_count;

    Vec3f evaluate(const float& u, const float& v, const float& w, const uint& prim, Resources& resources) const;

    ~GeometryMeshAttribute()
    {
        delete array;
        delete indices;
    }
};

class GeometryMesh : public Geometry
{
public:
    GeometryMesh(const Transform3f& obj_to_world, const std::string& filename);

    const GeometryAttribute * get_geometry_attribute(const std::string& name);

    ~GeometryMesh();

private:
    std::unordered_map<std::string, GeometryMeshAttribute*> mesh_attributes;
};

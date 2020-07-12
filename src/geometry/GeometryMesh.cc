#include <geometry/GeometryMesh.h>

#include <import/MeshFile.h>

GeometryMesh::GeometryMesh(const std::string& filename, const Transform3f& obj_to_world, const GeometryVisibility& visibility)
: Geometry(obj_to_world, visibility)
{
    // TODO: Check the filetype somehow
    std::vector<GeometryMeshAttribute*> read_attributes = MeshFile::ReadOBJ(filename);
    for(auto attr = read_attributes.begin(); attr != read_attributes.end(); ++attr)
        mesh_attributes[(*attr)->name] = *attr;

    const GeometryMeshAttribute * vertices = (const GeometryMeshAttribute *)get_geometry_attribute("vertices");
    if(!vertices) return;
    obj_bbox = Box3f();
    for(uint i = 0; i < vertices->array_item_count; i++)
        obj_bbox.extend(Vec3f(vertices->array[i*vertices->array_item_size + 0], 
                              vertices->array[i*vertices->array_item_size + 1], 
                              vertices->array[i*vertices->array_item_size + 2]));
    world_bbox = obj_to_world * obj_bbox;
}

const GeometryAttribute * GeometryMesh::get_geometry_attribute(const std::string& name)
{
    auto attribute = mesh_attributes.find(name);
    return (attribute != mesh_attributes.end()) ? attribute->second : nullptr;
}

Vec3f GeometryMeshAttribute::evaluate(const float& u, const float& v, const float& w, const uint& prim, Resources& resources) const
{
    Vec3f out = VEC3F_ZERO;

    if(indices_item_size - indices_item_pad == 3)
    {
        const uint index = indices_item_size*prim;
        const uint v0 = indices[index+0]*array_item_size;
        const uint v1 = indices[index+1]*array_item_size;
        const uint v2 = indices[index+2]*array_item_size;
        for(uint i = 0; i < array_item_size-array_item_pad; i++)
            out[i] = (1.f - u - v) * array[v0+i] + u * array[v1+i] + v * array[v2+i];
    }

    return out;
}

GeometryMesh::~GeometryMesh()
{
    for(auto attr = mesh_attributes.begin(); attr != mesh_attributes.end(); ++attr)
        delete attr->second;
}

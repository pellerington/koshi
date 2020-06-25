#include <geometry/GeometryMesh.h>

#include <import/MeshFile.h>

GeometryMesh::GeometryMesh(const Transform3f &obj_to_world, const std::string& filename)
: Geometry(obj_to_world)
{
    // TODO: Check the filetype somehow
    std::vector<GeometryMeshAttribute*> read_attributes = MeshFile::ReadOBJ(filename);
    for(auto attr = read_attributes.begin(); attr != read_attributes.end(); ++attr)
        mesh_attributes[(*attr)->name] = *attr;

    const GeometryMeshAttribute * vertices = get_mesh_attribute("vertices");
    if(!vertices) return;
    obj_bbox = Box3f();
    for(uint i = 0; i < vertices->array_item_count; i++)
        obj_bbox.extend(Vec3f(vertices->array[i*vertices->array_item_size + 0], 
                              vertices->array[i*vertices->array_item_size + 1], 
                              vertices->array[i*vertices->array_item_size + 2]));
    world_bbox = obj_to_world * obj_bbox;
}

const GeometryMeshAttribute * GeometryMesh::get_mesh_attribute(const std::string& name)
{
    auto attribute = mesh_attributes.find(name);
    return (attribute != mesh_attributes.end()) ? attribute->second : nullptr;
}

bool GeometryMesh::eval_geometry_attribute(Vec3f& out, const std::string& attribute_name, const float& u, const float& v, const float& w, const uint& prim, Resources& resources)
{
    auto item = mesh_attributes.find(attribute_name);
    if(item == mesh_attributes.end()) return false;

    const GeometryMeshAttribute * attr = item->second;

    if(attr->indices_item_size - attr->indices_item_pad == 3)
    {
        out = VEC3F_ZERO;
        const uint index = attr->indices_item_size*prim;
        const uint v0 = attr->indices[index+0]*attr->array_item_size;
        const uint v1 = attr->indices[index+1]*attr->array_item_size;
        const uint v2 = attr->indices[index+2]*attr->array_item_size;
        for(uint i = 0; i < attr->array_item_size-attr->array_item_pad; i++)
            out[i] = (1.f - u - v) * attr->array[v0+i] + u * attr->array[v1+i] + v * attr->array[v2+i];
        return true;
    }

    return false;
}

GeometryMesh::~GeometryMesh()
{
    for(auto attr = mesh_attributes.begin(); attr != mesh_attributes.end(); ++attr)
    {
        delete attr->second->array;
        delete attr->second->indices;
        delete attr->second;
    }
}

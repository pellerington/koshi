#include <geometry/GeometryMesh.h>

#include <import/MeshFile.h>

GeometryMesh::GeometryMesh(const Transform3f &obj_to_world, const std::string& filename)
: Geometry(obj_to_world)
{
    // TODO: Check the filetype somehow
    std::vector<GeometryMeshAttribute*> read_attributes = MeshFile::ReadOBJ(filename);
    for(auto attr = read_attributes.begin(); attr != read_attributes.end(); ++attr)
        attributes[(*attr)->name] = *attr;

    const GeometryMeshAttribute * vertices = get_mesh_attribute("vertices");
    if(!vertices) return;
    // TODO: Support float 3 as well, or find better way of doing it.
    float4 * vertices_array = (float4*)vertices->array;
    obj_bbox = Box3f();
    for(uint i = 0; i < vertices->array_size; i++)
        obj_bbox.extend(Vec3f(vertices_array[i].v0, vertices_array[i].v1, vertices_array[i].v2));
    world_bbox = obj_to_world * obj_bbox;
}

GeometryMesh::~GeometryMesh()
{
    for(auto attr = attributes.begin(); attr != attributes.end(); ++attr)
    {
        delete attr->second->array; // <- THIS DOESNT WORK???
        delete attr->second->indices;
        delete attr->second;
    }
}

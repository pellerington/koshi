#pragma once

#include "import/SceneFile.h"

#include <geometry/GeometryMesh.h>
#include <geometry/GeometryBox.h>
#include <geometry/GeometrySphere.h>

// TODO: Add defaults so we can add these in seperatly.
#include <embree/EmbreeGeometryMesh.h>
#include <embree/EmbreeGeometryBox.h>
#include <embree/EmbreeGeometrySphere.h>

#include <import/MeshFile.h>

struct GeometrySceneFile
{

    static void set_geometry_type(Type& type)
    {
        type.reserved_attributes.push_back("scale");
        type.reserved_attributes.push_back("rotation");
        type.reserved_attributes.push_back("translation");
    }

    static void add_types(Types& types)
    {
        // Geometry Mesh
        Type geometry_mesh("geometry_mesh");
        set_geometry_type(geometry_mesh);
        geometry_mesh.reserved_attributes.push_back("filetype");
        geometry_mesh.reserved_attributes.push_back("filename");
        geometry_mesh.create_object_cb = create_geometry_mesh;
        types.add(geometry_mesh);

        // Geometry Box
        Type geometry_box("geometry_box");
        set_geometry_type(geometry_box);
        geometry_box.create_object_cb = create_geometry_box;
        types.add(geometry_box);

        // Geometry Sphere
        Type geometry_sphere("geometry_sphere");
        set_geometry_type(geometry_sphere);
        geometry_sphere.create_object_cb = create_geometry_sphere;
        types.add(geometry_sphere);
    }

    static Transform3f get_transform(AttributeAccessor& accessor)
    {
        const Vec3f scale = accessor.get_vec3f("scale", 1.f);
        const Vec3f rotation = 2.f * PI * accessor.get_vec3f("rotation") / 360.f;
        const Vec3f translation = accessor.get_vec3f("translation");

        Transform3f transform;
        transform = transform * Transform3f::translation(translation);
        transform = transform * Transform3f::z_rotation(rotation.z);
        transform = transform * Transform3f::y_rotation(rotation.y);
        transform = transform * Transform3f::x_rotation(rotation.x);
        transform = transform * Transform3f::scale(scale);

        return transform;
    }

    static Object * create_geometry_mesh(AttributeAccessor& accessor, Object * parent)
    {
        Transform3f transform = get_transform(accessor);
        GeometryMesh * geometry = nullptr;

        std::string filetype = accessor.get_string("filetype");
        if(filetype == "obj")
            geometry = MeshFile::ImportOBJ(accessor.get_string("filename"), transform);

        // TODO: Move embree geometry somewhere else.
        if(geometry) {
            EmbreeGeometryMesh * embree = new EmbreeGeometryMesh(geometry);
            geometry->set_attribute("embree_geometry", embree);
            accessor.add_object("embree_geometry", embree);
        }

        return geometry;
    }

    static Object * create_geometry_box(AttributeAccessor& accessor, Object * parent)
    {
        Transform3f transform = get_transform(accessor);
        GeometryBox * geometry = new GeometryBox(transform);
        EmbreeGeometryBox * embree = new EmbreeGeometryBox(geometry);
        geometry->set_attribute("embree_geometry", embree);
        accessor.add_object("embree_geometry", embree);
        return geometry;
    }

    static Object * create_geometry_sphere(AttributeAccessor& accessor, Object * parent)
    {
        Transform3f transform = get_transform(accessor);
        GeometrySphere * geometry = new GeometrySphere(transform);
        EmbreeGeometrySphere * embree = new EmbreeGeometrySphere(geometry);
        geometry->set_attribute("embree_geometry", embree);
        accessor.add_object("embree_geometry", embree);
        return geometry;
    }
};
#pragma once

#include "SceneFile.h"

#include <koshi/geometry/GeometryMesh.h>
#include <koshi/geometry/GeometryBox.h>
#include <koshi/geometry/GeometrySphere.h>
#include <koshi/geometry/GeometryArea.h>
#include <koshi/geometry/GeometryVolume.h>

// TODO: Add defaults so we can add these in seperatly.
#include <koshi/embree/EmbreeGeometryMesh.h>
#include <koshi/embree/EmbreeGeometryBox.h>
#include <koshi/embree/EmbreeGeometrySphere.h>
#include <koshi/embree/EmbreeGeometryArea.h>
#include <koshi/embree/EmbreeGeometryVolume.h>

struct GeometrySceneFile
{

    static void set_geometry_type(Type& type)
    {
        type.reserved_attributes.push_back("scale");
        type.reserved_attributes.push_back("rotation");
        type.reserved_attributes.push_back("translation");
        type.reserved_attributes.push_back("hide_camera");
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

        // Geometry Area
        Type geometry_area("geometry_area");
        set_geometry_type(geometry_area);
        geometry_area.create_object_cb = create_geometry_area;
        types.add(geometry_area);

        // Geometry Volume
        Type geometry_volume("geometry_volume");
        set_geometry_type(geometry_volume);
        geometry_volume.create_object_cb = create_geometry_volume;
        types.add(geometry_volume);
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

    static GeometryVisibility get_visibility(AttributeAccessor& accessor)
    {
        GeometryVisibility visibility;
        visibility.hide_camera = accessor.get_bool("hide_camera");
        return visibility;
    }

    static Object * create_geometry_mesh(AttributeAccessor& accessor, Object * parent)
    {
        std::string filetype = accessor.get_string("filetype");
        std::string filename = accessor.get_string("filename");

        Transform3f transform = get_transform(accessor);
        GeometryVisibility visibility = get_visibility(accessor);

        GeometryMesh * geometry = new GeometryMesh(filename, transform, visibility);

        EmbreeGeometryMesh * embree = new EmbreeGeometryMesh(geometry);
        geometry->set_attribute("embree_geometry", embree);
        accessor.add_object("embree_geometry", embree);

        return geometry;
    }

    static Object * create_geometry_box(AttributeAccessor& accessor, Object * parent)
    {
        Transform3f transform = get_transform(accessor);
        GeometryVisibility visibility = get_visibility(accessor);

        GeometryBox * geometry = new GeometryBox(transform, visibility);

        EmbreeGeometryBox * embree = new EmbreeGeometryBox(geometry);
        geometry->set_attribute("embree_geometry", embree);
        accessor.add_object("embree_geometry", embree);

        return geometry;
    }

    static Object * create_geometry_sphere(AttributeAccessor& accessor, Object * parent)
    {
        Transform3f transform = get_transform(accessor);
        GeometryVisibility visibility = get_visibility(accessor);

        GeometrySphere * geometry = new GeometrySphere(transform, visibility);

        EmbreeGeometrySphere * embree = new EmbreeGeometrySphere(geometry);
        geometry->set_attribute("embree_geometry", embree);
        accessor.add_object("embree_geometry", embree);

        return geometry;
    }

    static Object * create_geometry_area(AttributeAccessor& accessor, Object * parent)
    {
        Transform3f transform = get_transform(accessor);
        GeometryVisibility visibility = get_visibility(accessor);

        GeometryArea * geometry = new GeometryArea(transform, visibility);

        EmbreeGeometryArea * embree = new EmbreeGeometryArea(geometry);
        geometry->set_attribute("embree_geometry", embree);
        accessor.add_object("embree_geometry", embree);

        return geometry;
    }

    static Object * create_geometry_volume(AttributeAccessor& accessor, Object * parent)
    {
        Transform3f transform = get_transform(accessor);
        GeometryVisibility visibility = get_visibility(accessor);

        GeometryVolume * geometry = new GeometryVolume(transform, visibility);

        EmbreeGeometryVolume * embree = new EmbreeGeometryVolume(geometry);
        geometry->set_attribute("embree_geometry", embree);
        accessor.add_object("embree_geometry", embree);

        return geometry;
    }
};
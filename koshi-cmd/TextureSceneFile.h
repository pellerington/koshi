#pragma once

#include "SceneFile.h"

#include <koshi/texture/TextureImage.h>
#include <koshi/texture/TextureChecker.h>
#include <koshi/texture/TextureGrid.h>
#include <koshi/texture/TextureGradient.h>
#include <koshi/texture/TextureOpenVDB.h>
#include <koshi/texture/TextureGeometryAttribute.h>
#include <koshi/texture/TextureConstant.h>

struct TextureSceneFile
{
    static void add_types(Types& types)
    {
        // Texture Constant
        Type texture_constant("texture_constant");
        texture_constant.reserved_attributes.push_back("color");
        texture_constant.create_object_cb = create_texture_constant;
        types.add(texture_constant);

        // Texture Image
        Type texture_image("texture_image");
        texture_image.reserved_attributes.push_back("filename");
        texture_image.reserved_attributes.push_back("smooth");
        texture_image.create_object_cb = create_texture_image;
        types.add(texture_image);

        // Texture Checker
        Type texture_checker("texture_checker");
        texture_checker.reserved_attributes.push_back("scale");
        texture_checker.create_object_cb = create_texture_checker;
        types.add(texture_checker);

        // Texture Grid
        Type texture_grid("texture_grid");
        texture_grid.reserved_attributes.push_back("fill");
        texture_grid.reserved_attributes.push_back("line");
        texture_grid.reserved_attributes.push_back("line_size");
        texture_grid.reserved_attributes.push_back("scale");
        texture_grid.create_object_cb = create_texture_grid;
        types.add(texture_grid);

        // Texture Gradient
        Type texture_gradient("texture_gradient");
        texture_gradient.reserved_attributes.push_back("min");
        texture_gradient.reserved_attributes.push_back("max");
        texture_gradient.reserved_attributes.push_back("axis");
        texture_gradient.create_object_cb = create_texture_gradient;
        types.add(texture_gradient);

        // Texture Openvdb
        Type texture_openvdb("texture_openvdb");
        texture_openvdb.reserved_attributes.push_back("filename");
        texture_openvdb.reserved_attributes.push_back("gridname");
        texture_openvdb.create_object_cb = create_texture_openvdb;
        types.add(texture_openvdb);

        // Texture GeometryAttribute
        Type texture_geometry_attribute("texture_geometry_attribute");
        texture_geometry_attribute.reserved_attributes.push_back("attribute");
        texture_geometry_attribute.create_object_cb = create_texture_geometry_attribute;
        types.add(texture_geometry_attribute);
    }

    static Object * create_texture_constant(AttributeAccessor& accessor, Object * parent)
    {
        const Vec3f color = accessor.get_vec3f("color");
        return new TextureConstant(color);
    }

    static Object * create_texture_image(AttributeAccessor& accessor, Object * parent)
    {
        const std::string filename = accessor.get_string("filename");
        const bool smooth = accessor.get_bool("smooth", true);
        return new TextureImage(filename, smooth);
    }

    static Object * create_texture_checker(AttributeAccessor& accessor, Object * parent)
    {
        const Vec3f scale = accessor.get_vec3f("scale", 1.f);
        return new TextureChecker(scale);
    }

    static Object * create_texture_grid(AttributeAccessor& accessor, Object * parent)
    {
        const Vec3f fill = accessor.get_vec3f("fill", 1.f);
        const Vec3f line = accessor.get_vec3f("line", 0.f);
        const float line_size = accessor.get_float("line_size", 0.1f);
        const Vec3f scale = accessor.get_vec3f("scale", 1.f);
        return new TextureGrid(fill, line, line_size, scale);
    }

    static Object * create_texture_gradient(AttributeAccessor& accessor, Object * parent)
    {
        const Vec3f min = accessor.get_vec3f("min", 0.f);
        const Vec3f max = accessor.get_vec3f("max", 1.f);
        const uint axis = accessor.get_uint("axis", 0);
        return new TextureGradient(min, max, axis);
    }

    static Object * create_texture_openvdb(AttributeAccessor& accessor, Object * parent)
    {
        const std::string filename = accessor.get_string("filename");
        const std::string gridname = accessor.get_string("gridname");
        return new TexutreOpenVDB(filename, gridname);
    }

    static Object * create_texture_geometry_attribute(AttributeAccessor& accessor, Object * parent)
    {
        const std::string attribute = accessor.get_string("attribute");
        return new TextureGeometryAttribute(attribute);
    }

};

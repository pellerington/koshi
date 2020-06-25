#pragma once

#include <import/SceneFile.h>

#include <texture/TextureImage.h>
#include <texture/TextureChecker.h>
#include <texture/TextureGradient.h>
#include <texture/TextureOpenVDB.h>

struct TextureSceneFile
{
    static void add_types(Types& types)
    {
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

};

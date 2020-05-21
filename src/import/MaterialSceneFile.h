#pragma once

#include "import/SceneFile.h"

#include "materials/MaterialLambert.h"
#include "materials/MaterialGGXReflect.h"
#include "materials/MaterialGGXRefract.h"
#include "materials/MaterialDielectric.h"

struct MaterialSceneFile
{

    static void add_types(Types& types)
    {
        // Material Lambert
        Type material_lambert("material_lambert");
        material_lambert.reserved_attributes.push_back("color");
        material_lambert.reserved_attributes.push_back("color_texture");
        material_lambert.create_object_cb = create_material_lambert;
        types.add(material_lambert);

        // TODO: Material BackLambert

        // Material GGX
        Type material_ggx("material_ggx");
        material_ggx.reserved_attributes.push_back("color");
        material_ggx.reserved_attributes.push_back("color_texture");
        material_ggx.reserved_attributes.push_back("roughness");
        material_ggx.reserved_attributes.push_back("roughness_texture");
        material_ggx.create_object_cb = create_material_ggx;
        types.add(material_ggx);

        // Material GGX Refract
        Type material_ggx_refract("material_ggx_refract");
        material_ggx_refract.reserved_attributes.push_back("color");
        material_ggx_refract.reserved_attributes.push_back("color_texture");
        material_ggx_refract.reserved_attributes.push_back("color_depth");
        material_ggx_refract.reserved_attributes.push_back("roughness");
        material_ggx_refract.reserved_attributes.push_back("roughness_texture");
        material_ggx_refract.reserved_attributes.push_back("ior");
        material_ggx_refract.create_object_cb = create_material_ggx_refract;
        types.add(material_ggx_refract);

        // Material Dielectric
        Type material_dielectric("material_dielectric");
        material_dielectric.reserved_attributes.push_back("reflect_color");
        material_dielectric.reserved_attributes.push_back("reflect_color_texture");
        material_dielectric.reserved_attributes.push_back("refract_color");
        material_dielectric.reserved_attributes.push_back("refract_color_texture");
        material_dielectric.reserved_attributes.push_back("refract_color_depth");
        material_dielectric.reserved_attributes.push_back("roughness");
        material_dielectric.reserved_attributes.push_back("roughness_texture");
        material_dielectric.reserved_attributes.push_back("ior");
        material_dielectric.create_object_cb = create_material_dielectric;
        types.add(material_dielectric);
    }

    static Object * create_material_lambert(AttributeAccessor& accessor, Object * parent)
    {
        const Vec3f color = accessor.get_vec3f("color");
        Texture * color_texture = dynamic_cast<Texture*>(accessor.get_object("color_texture"));

        return new MaterialLambert<true>(AttributeVec3f(color_texture, color));
    }

    //TODO : Combine lambert and back lambert

    static Object * create_material_ggx(AttributeAccessor& accessor, Object * parent)
    {
        const Vec3f color = accessor.get_vec3f("color");
        Texture * color_texture = dynamic_cast<Texture*>(accessor.get_object("color_texture"));
        const float roughness = accessor.get_float("roughness");
        Texture * roughness_texture = dynamic_cast<Texture*>(accessor.get_object("roughness_texture"));

        return new MaterialGGXReflect(
            AttributeVec3f(color_texture, color),
            AttributeFloat(roughness_texture, roughness)
        );
    }

    static Object * create_material_ggx_refract(AttributeAccessor& accessor, Object * parent)
    {
        const Vec3f color = accessor.get_vec3f("color");
        Texture * color_texture = dynamic_cast<Texture*>(accessor.get_object("color_texture"));
        const float color_depth = accessor.get_float("color_depth");
        const float roughness = accessor.get_float("roughness");
        Texture * roughness_texture = dynamic_cast<Texture*>(accessor.get_object("roughness_texture"));
        const float ior = accessor.get_float("ior");

        return new MaterialGGXRefract(
            AttributeVec3f(color_texture, color),
            AttributeFloat(roughness_texture, roughness), 
            ior, color_depth
        );
    }

    static Object * create_material_dielectric(AttributeAccessor& accessor, Object * parent)
    {
        const Vec3f reflect_color = accessor.get_vec3f("reflect_color");
        Texture * reflect_color_texture = dynamic_cast<Texture*>(accessor.get_object("reflect_color_texture"));
        const Vec3f refract_color = accessor.get_vec3f("refract_color");
        Texture * refract_color_texture = dynamic_cast<Texture*>(accessor.get_object("refract_color_texture"));
        const float refract_color_depth = accessor.get_float("refract_color_depth");
        const float roughness = accessor.get_float("roughness");
        Texture * roughness_texture = dynamic_cast<Texture*>(accessor.get_object("roughness_texture"));
        const float ior = accessor.get_float("ior");

        return new MaterialDielectric(
            AttributeVec3f(reflect_color_texture, reflect_color),
            AttributeVec3f(refract_color_texture, refract_color),
            refract_color_depth,
            AttributeFloat(roughness_texture, roughness), ior
        );
    }
};
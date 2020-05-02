#pragma once

#include "import/SceneFile.h"

#include "Materials/MaterialLambert.h"
#include "Materials/MaterialGGXReflect.h"

struct MaterialSceneFile
{

    static void add_types(Types& types)
    {
        // Material Lambert
        Type material_lambert("material_lambert");
        material_lambert.reserved_attributes.push_back("diffuse_color");
        material_lambert.reserved_attributes.push_back("diffuse_color_texture");
        material_lambert.create_object_cb = create_material_lambert;
        types.add(material_lambert);

        // TODO: Material BackLambert

        // Material GGX
        Type material_ggx("material_ggx");
        material_ggx.reserved_attributes.push_back("specular_color");
        material_ggx.reserved_attributes.push_back("specular_color_texture");
        material_ggx.reserved_attributes.push_back("roughness");
        material_ggx.reserved_attributes.push_back("roughness_texture");
        material_ggx.create_object_cb = create_material_ggx;
        types.add(material_ggx);
    }

    static Object * create_material_lambert(AttributeAccessor& accessor, Object * parent)
    {
        const Vec3f diffuse_color = accessor.get_vec3f("diffuse_color");
        Texture * diffuse_color_texture = dynamic_cast<Texture*>(accessor.get_object("diffuse_color_texture"));

        return new MaterialLambert(AttributeVec3f(diffuse_color_texture, diffuse_color));
    }

    //TODO : Combine lambert and back lambert

    static Object * create_material_ggx(AttributeAccessor& accessor, Object * parent)
    {
        const Vec3f specular_color = accessor.get_vec3f("specular_color");
        Texture * specular_color_texture = dynamic_cast<Texture*>(accessor.get_object("specular_color_texture"));
        const float roughness = accessor.get_float("roughness");
        Texture * roughness_texture = dynamic_cast<Texture*>(accessor.get_object("roughness_texture"));

        return new MaterialGGXReflect(
            AttributeVec3f(specular_color_texture, specular_color),
            AttributeFloat(roughness_texture, roughness)
        );
    }

};

    //                 if((*it)["type"] == "ggx_refract")
    //                 {
    //                     const Vec3f refractive_color = get_vec3f(*it, "refractive_color");
    //                     Texture * refractive_color_texture = ((*it)["refractive_color_texture"].is_string()) ? textures[(*it)["refractive_color_texture"]] : nullptr;

    //                     const float roughness = get_float(*it, "roughness");
    //                     Texture * roughness_texture = ((*it)["roughness_texture"].is_string()) ? textures[(*it)["roughness_texture"]] : nullptr;

    //                     const float ior = get_float(*it, "ior", 1.f);

    //                     Material * material = new MaterialGGXRefract(
    //                         AttributeVec3f(refractive_color_texture, refractive_color),
    //                         AttributeFloat(roughness_texture, roughness), ior
    //                     );
    //                     materials[(*it)["name"]] = material;
    //                     scene.add_object(material);
    //                 }

    //                 if((*it)["type"] == "dielectric")
    //                 {
    //                     const Vec3f reflective_color = get_vec3f(*it, "reflective_color");
    //                     Texture * reflective_color_texture = ((*it)["reflective_color_texture"].is_string()) ? textures[(*it)["reflective_color_texture"]] : nullptr;

    //                     const Vec3f refractive_color = get_vec3f(*it, "refractive_color");
    //                     Texture * refractive_color_texture = ((*it)["refractive_color_texture"].is_string()) ? textures[(*it)["refractive_color_texture"]] : nullptr;

    //                     const float roughness = get_float(*it, "roughness");
    //                     Texture * roughness_texture = ((*it)["roughness_texture"].is_string()) ? textures[(*it)["roughness_texture"]] : nullptr;

    //                     const float ior = get_float(*it, "ior", 1.f);

    //                     Material * material = new MaterialDielectric(
    //                         AttributeVec3f(reflective_color_texture, reflective_color),
    //                         AttributeVec3f(refractive_color_texture, refractive_color),
    //                         AttributeFloat(roughness_texture, roughness),
    //                         ior
    //                     );
    //                     materials[(*it)["name"]] = material;
    //                     scene.add_object(material);
    //                 }
#pragma once

#include "SceneFile.h"

#include <koshi/material/MaterialLambert.h>
#include <koshi/material/MaterialGGXReflect.h>
#include <koshi/material/MaterialGGXRefract.h>
#include <koshi/material/MaterialDielectric.h>
#include <koshi/material/MaterialRandomWalk.h>
#include <koshi/material/MaterialVolume.h>

struct MaterialSceneFile
{

    static void add_types(Types& types)
    {
        // Material Lambert
        Type material_lambert("material_lambert");
        material_lambert.reserved_attributes.push_back("color");
        material_lambert.reserved_attributes.push_back("color_texture");
        material_lambert.reserved_attributes.push_back("normal_texture");
        material_lambert.create_object_cb = create_material_lambert;
        types.add(material_lambert);

        // TODO: Material BackLambert

        // Material GGX
        Type material_ggx("material_ggx");
        material_ggx.reserved_attributes.push_back("color");
        material_ggx.reserved_attributes.push_back("color_texture");
        material_ggx.reserved_attributes.push_back("roughness");
        material_ggx.reserved_attributes.push_back("roughness_texture");
        material_ggx.reserved_attributes.push_back("normal_texture");
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
        material_ggx_refract.reserved_attributes.push_back("normal_texture");
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
        material_dielectric.reserved_attributes.push_back("normal_texture");
        material_dielectric.create_object_cb = create_material_dielectric;
        types.add(material_dielectric);

        // Material Random Walk
        Type material_randomwalk("material_randomwalk");
        material_randomwalk.reserved_attributes.push_back("color");
        material_randomwalk.reserved_attributes.push_back("color_texture");
        material_randomwalk.reserved_attributes.push_back("density");
        material_randomwalk.reserved_attributes.push_back("transparency");
        material_randomwalk.reserved_attributes.push_back("density_texture");
        material_randomwalk.reserved_attributes.push_back("anistropy");
        material_randomwalk.reserved_attributes.push_back("normal_texture");
        material_randomwalk.create_object_cb = create_material_randomwalk;
        types.add(material_randomwalk);

        // Material Volume
        Type material_volume("material_volume");
        material_volume.reserved_attributes.push_back("density");
        material_volume.reserved_attributes.push_back("density_texture");
        material_volume.reserved_attributes.push_back("density_attribute");
        material_volume.reserved_attributes.push_back("scatter");
        material_volume.reserved_attributes.push_back("scatter_texture");
        material_volume.reserved_attributes.push_back("scatter_attribute");
        material_volume.reserved_attributes.push_back("emission");
        material_volume.reserved_attributes.push_back("emission_texture");
        material_volume.reserved_attributes.push_back("emission_attribute");
        material_volume.reserved_attributes.push_back("anistropy");
        material_volume.reserved_attributes.push_back("anistropy_texture");
        material_volume.create_object_cb = create_material_volume;
        types.add(material_volume);
    }

    static Object * create_material_lambert(AttributeAccessor& accessor, Object * parent)
    {
        const Vec3f color = accessor.get_vec3f("color");
        Texture * color_texture = dynamic_cast<Texture*>(accessor.get_object("color_texture"));
        TextureMultiply * color_multiply_texture = new TextureMultiply(color, color_texture);
        accessor.add_object("color_multiply_texture", color_multiply_texture);

        Texture * normal_texture = dynamic_cast<Texture*>(accessor.get_object("normal_texture"));

        const Vec3f opacity = accessor.get_vec3f("opacity", 1.f);
        Texture * opacity_texture = dynamic_cast<Texture*>(accessor.get_object("opacity_texture"));
        TextureMultiply * opacity_multiply_texture = new TextureMultiply(opacity, opacity_texture);
        accessor.add_object("opacity_multiply_texture", opacity_multiply_texture);

        return new MaterialLambert<true>(color_multiply_texture, normal_texture, opacity_multiply_texture);
    }

    static Object * create_material_ggx(AttributeAccessor& accessor, Object * parent)
    {
        const Vec3f color = accessor.get_vec3f("color");
        Texture * color_texture = dynamic_cast<Texture*>(accessor.get_object("color_texture"));
        TextureMultiply * color_multiply_texture = new TextureMultiply(color, color_texture);
        accessor.add_object("color_multiply_texture", color_multiply_texture);

        const float roughness = accessor.get_float("roughness");
        Texture * roughness_texture = dynamic_cast<Texture*>(accessor.get_object("roughness_texture"));
        TextureMultiply * roughness_multiply_texture = new TextureMultiply(roughness, roughness_texture);
        accessor.add_object("roughness_multiply_texture", roughness_multiply_texture);

        Texture * normal_texture = dynamic_cast<Texture*>(accessor.get_object("normal_texture"));

        const Vec3f opacity = accessor.get_vec3f("opacity", 1.f);
        Texture * opacity_texture = dynamic_cast<Texture*>(accessor.get_object("opacity_texture"));
        TextureMultiply * opacity_multiply_texture = new TextureMultiply(opacity, opacity_texture);
        accessor.add_object("opacity_multiply_texture", opacity_multiply_texture);

        return new MaterialGGXReflect(color_multiply_texture, roughness_multiply_texture, 
                                      normal_texture, opacity_multiply_texture);
    }

    static Object * create_material_ggx_refract(AttributeAccessor& accessor, Object * parent)
    {
        const Vec3f color = accessor.get_vec3f("color");
        Texture * color_texture = dynamic_cast<Texture*>(accessor.get_object("color_texture"));
        TextureMultiply * color_multiply_texture = new TextureMultiply(color, color_texture);
        accessor.add_object("color_multiply_texture", color_multiply_texture);

        const float color_depth = accessor.get_float("color_depth");

        const float roughness = accessor.get_float("roughness");
        Texture * roughness_texture = dynamic_cast<Texture*>(accessor.get_object("roughness_texture"));
        TextureMultiply * roughness_multiply_texture = new TextureMultiply(roughness, roughness_texture);
        accessor.add_object("roughness_multiply_texture", roughness_multiply_texture);

        const float ior = accessor.get_float("ior");

        Texture * normal_texture = dynamic_cast<Texture*>(accessor.get_object("normal_texture"));

        const Vec3f opacity = accessor.get_vec3f("opacity", 1.f);
        Texture * opacity_texture = dynamic_cast<Texture*>(accessor.get_object("opacity_texture"));
        TextureMultiply * opacity_multiply_texture = new TextureMultiply(opacity, opacity_texture);
        accessor.add_object("opacity_multiply_texture", opacity_multiply_texture);

        return new MaterialGGXRefract(
            color_multiply_texture, roughness_multiply_texture, 
            ior, color_depth, normal_texture, opacity_multiply_texture
        );
    }

    static Object * create_material_dielectric(AttributeAccessor& accessor, Object * parent)
    {
        const Vec3f reflect_color = accessor.get_vec3f("reflect_color");
        Texture * reflect_color_texture = dynamic_cast<Texture*>(accessor.get_object("reflect_color_texture"));
        TextureMultiply * reflect_color_multiply_texture = new TextureMultiply(reflect_color, reflect_color_texture);
        accessor.add_object("reflect_color_multiply_texture", reflect_color_multiply_texture);

        const Vec3f refract_color = accessor.get_vec3f("refract_color");
        Texture * refract_color_texture = dynamic_cast<Texture*>(accessor.get_object("refract_color_texture"));
        TextureMultiply * refract_color_multiply_texture = new TextureMultiply(refract_color, refract_color_texture);
        accessor.add_object("refract_color_multiply_texture", refract_color_multiply_texture);
  
        const float refract_color_depth = accessor.get_float("refract_color_depth");
        
        const float roughness = accessor.get_float("roughness");
        Texture * roughness_texture = dynamic_cast<Texture*>(accessor.get_object("roughness_texture"));
        TextureMultiply * roughness_multiply_texture = new TextureMultiply(roughness, roughness_texture);
        accessor.add_object("roughness_multiply_texture", roughness_multiply_texture);

        const float ior = accessor.get_float("ior");

        Texture * normal_texture = dynamic_cast<Texture*>(accessor.get_object("normal_texture"));

        const Vec3f opacity = accessor.get_vec3f("opacity", 1.f);
        Texture * opacity_texture = dynamic_cast<Texture*>(accessor.get_object("opacity_texture"));
        TextureMultiply * opacity_multiply_texture = new TextureMultiply(opacity, opacity_texture);
        accessor.add_object("opacity_multiply_texture", opacity_multiply_texture);

        return new MaterialDielectric(
            reflect_color_multiply_texture, refract_color_multiply_texture,
            refract_color_depth, roughness_multiply_texture, ior,
            normal_texture, opacity_multiply_texture
        );
    }

    static Object * create_material_randomwalk(AttributeAccessor& accessor, Object * parent)
    {
        const Vec3f color = accessor.get_vec3f("color");
        Texture * color_texture = dynamic_cast<Texture*>(accessor.get_object("color_texture"));
        TextureMultiply * color_multiply_texture = new TextureMultiply(color, color_texture);
        accessor.add_object("color_multiply_texture", color_multiply_texture);

        Vec3f density = accessor.get_float("density");
        density *= VEC3F_ONES - accessor.get_vec3f("transparency");
        Texture * density_texture = dynamic_cast<Texture*>(accessor.get_object("density_texture"));
        TextureMultiply * density_multiply_texture = new TextureMultiply(density, density_texture);
        accessor.add_object("density_multiply_texture", density_multiply_texture);

        const float anistropy = accessor.get_float("anistropy");

        Texture * normal_texture = dynamic_cast<Texture*>(accessor.get_object("normal_texture"));

        const Vec3f opacity = accessor.get_vec3f("opacity", 1.f);
        Texture * opacity_texture = dynamic_cast<Texture*>(accessor.get_object("opacity_texture"));
        TextureMultiply * opacity_multiply_texture = new TextureMultiply(opacity, opacity_texture);
        accessor.add_object("opacity_multiply_texture", opacity_multiply_texture);

        return new MaterialRandomWalk(color_multiply_texture, density_multiply_texture, anistropy, 
                                      normal_texture, opacity_multiply_texture);      
    }

    static Object * create_material_volume(AttributeAccessor& accessor, Object * parent)
    {
        const Vec3f density = accessor.get_vec3f("density");
        Texture * density_texture = dynamic_cast<Texture*>(accessor.get_object("density_texture"));
        TextureMultiply * density_multiply_texture = new TextureMultiply(density, density_texture);
        accessor.add_object("density_multiply_texture", density_multiply_texture);
        const std::string density_attribute = accessor.get_string("density_attribute");

        const Vec3f scatter = accessor.get_vec3f("scatter");
        Texture * scatter_texture = dynamic_cast<Texture*>(accessor.get_object("scatter_texture"));
        TextureMultiply * scatter_multiply_texture = new TextureMultiply(scatter, scatter_texture);
        accessor.add_object("scatter_multiply_texture", scatter_multiply_texture);
        const std::string scatter_attribute = accessor.get_string("scatter_attribute");

        const Vec3f emission = accessor.get_vec3f("emission");
        Texture * emission_texture = dynamic_cast<Texture*>(accessor.get_object("emission_texture"));
        TextureMultiply * emission_multiply_texture = new TextureMultiply(emission, emission_texture);
        accessor.add_object("emission_multiply_texture", emission_multiply_texture);
        const std::string emission_attribute = accessor.get_string("emission_attribute");

        // TODO: This being a multiply texture is bad?
        const float anistropy = accessor.get_float("anistropy");
        Texture * anistropy_texture = dynamic_cast<Texture*>(accessor.get_object("anistropy_texture"));
        TextureMultiply * anistropy_multiply_texture = new TextureMultiply(anistropy, anistropy_texture);
        accessor.add_object("anistropy_multiply_texture", anistropy_multiply_texture);

        return new MaterialVolume(
            density_multiply_texture, density_attribute,
            scatter_multiply_texture, scatter_attribute,
            emission_multiply_texture, emission_attribute,
            anistropy_multiply_texture
        );
    }
};
#pragma once

#include "import/SceneFile.h"

#include <lights/Light.h>

#include <geometry/GeometryEnvironment.h>
#include <geometry/GeometryArea.h>
#include <geometry/GeometrySphere.h>

#include <lights/LightSamplerArea.h>
#include <lights/LightSamplerSphere.h>

#include <embree/EmbreeGeometryArea.h>
#include <embree/EmbreeGeometrySphere.h>

struct LightSceneFile
{

    static void set_light_type(Type& type)
    {
        type.reserved_attributes.push_back("scale");
        type.reserved_attributes.push_back("rotation");
        type.reserved_attributes.push_back("translation");

        type.reserved_attributes.push_back("intesity");
        type.reserved_attributes.push_back("intensity_texture");
    }

    static void add_types(Types& types)
    {
        // Light Environment
        Type light_environment("light_environment");
        set_light_type(light_environment);
        light_environment.create_object_cb = create_light_environment;
        types.add(light_environment);

        // Light Area
        Type light_area("light_area");
        set_light_type(light_area);
        light_area.create_object_cb = create_light_area;
        types.add(light_area);

        // Light Sphere
        Type light_sphere("light_sphere");
        set_light_type(light_sphere);
        light_sphere.create_object_cb = create_light_sphere;
        types.add(light_sphere);
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

    static Object * create_light_environment(AttributeAccessor& accessor, Object * parent)
    {
        GeometryEnvironment * environment = new GeometryEnvironment;

        const Vec3f intensity = accessor.get_vec3f("intensity");
        Texture * intensity_texture = dynamic_cast<Texture*>(accessor.get_object("intensity_texture"));
        Light * light = new Light(AttributeVec3f(intensity_texture, intensity));
        accessor.add_object("light", light);
        environment->set_attribute("light", light);

        return environment;
    }

    static Object * create_light_area(AttributeAccessor& accessor, Object * parent)
    {
        Transform3f transform = get_transform(accessor);
        GeometryArea * area_light = new GeometryArea(transform);

        const Vec3f intensity = accessor.get_vec3f("intensity");
        Texture * intensity_texture = dynamic_cast<Texture*>(accessor.get_object("intensity_texture"));
        Light * light = new Light(AttributeVec3f(intensity_texture, intensity));
        accessor.add_object("light", light);
        area_light->set_attribute("light", light);

        LightSamplerArea * light_sampler = new LightSamplerArea(area_light);
        accessor.add_object("light_sampler", light_sampler);
        area_light->set_attribute("light_sampler", light_sampler);

        // TODO: move this to some defaults.
        EmbreeGeometryArea * embree_geometry = new EmbreeGeometryArea(area_light);
        area_light->set_attribute("embree_geometry", embree_geometry);

        return area_light;
    }

    static Object * create_light_sphere(AttributeAccessor& accessor, Object * parent)
    {
        Transform3f transform = get_transform(accessor);
        GeometrySphere * sphere_light = new GeometrySphere(transform);

        const Vec3f intensity = accessor.get_vec3f("intensity");
        Texture * intensity_texture = dynamic_cast<Texture*>(accessor.get_object("intensity_texture"));
        Light * light = new Light(AttributeVec3f(intensity_texture, intensity));
        accessor.add_object("light", light);
        sphere_light->set_attribute("light", light);

        LightSamplerSphere * light_sampler = new LightSamplerSphere(sphere_light);
        accessor.add_object("light_sampler", light_sampler);
        sphere_light->set_attribute("light_sampler", light_sampler);

        // TODO: move this to some defaults.
        EmbreeGeometrySphere * embree_geometry = new EmbreeGeometrySphere(sphere_light);
        sphere_light->set_attribute("embree_geometry", embree_geometry);

        return sphere_light;
    }

};


    //             if((*it)["type"] == "sphere")
    //             {
    //                 const Vec3f scale = get_vec3f(*it, "scale", 1.f);
    //                 const Vec3f rotation = 2.f * PI * get_vec3f(*it, "rotation") / 360.f;
    //                 const Vec3f translation = get_vec3f(*it, "translation");

    //                 Transform3f transform;
    //                 transform = transform * Transform3f::translation(translation);
    //                 transform = transform * Transform3f::z_rotation(rotation.z);
    //                 transform = transform * Transform3f::y_rotation(rotation.y);
    //                 transform = transform * Transform3f::x_rotation(rotation.x);
    //                 transform = transform * Transform3f::scale(scale);

    //                 GeometrySphere * sphere = new GeometrySphere(transform);

    //                 const Vec3f intensity = get_vec3f(*it, "intensity");
    //                 Texture * intensity_texture = ((*it)["intensity_texture"].is_string()) ? textures[(*it)["intensity_texture"]] : nullptr;
    //                 // Todo: this is not destructed properly (needs to be added to the scene)
    //                 Light * light = new Light(AttributeVec3f(intensity_texture, intensity));

    //                 sphere->set_attribute("light", light);

    //                 EmbreeGeometrySphere * embree_geometry = new EmbreeGeometrySphere(sphere);
    //                 sphere->set_attribute("embree_geometry", embree_geometry);

    //                 LightSamplerSphere * light_sampler = new LightSamplerSphere(sphere);
    //                 sphere->set_attribute("light_sampler", light_sampler);

    //                 sphere->set_attribute("integrator", default_integrator);

    //                 scene.add_object(sphere);
    //             }
    //         }
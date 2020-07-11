#pragma once

#include "import/SceneFile.h"

#include <light/Light.h>

#include <import/GeometrySceneFile.h>

#include <geometry/GeometryEnvironment.h>

#include <light/LightSamplerArea.h>
#include <light/LightSamplerSphere.h>
#include <light/LightSamplerDirectional.h>

struct LightSceneFile
{

    static void set_light_type(Type& type)
    {
        type.reserved_attributes.push_back("intesity");
        type.reserved_attributes.push_back("intensity_texture");
    }

    static void add_types(Types& types)
    {
        // Light Environment
        Type light_environment("light_environment");
        GeometrySceneFile::set_geometry_type(light_environment);
        set_light_type(light_environment);
        light_environment.create_object_cb = create_light_environment;
        types.add(light_environment);

        // Light Environment
        Type light_directional("light_directional");
        GeometrySceneFile::set_geometry_type(light_directional);
        set_light_type(light_directional);
        light_directional.create_object_cb = create_light_directional;
        types.add(light_directional);

        // Light Area
        Type light_area("light_area");
        GeometrySceneFile::set_geometry_type(light_area);
        set_light_type(light_area);
        light_area.create_object_cb = create_light_area;
        types.add(light_area);

        // Light Sphere
        Type light_sphere("light_sphere");
        GeometrySceneFile::set_geometry_type(light_sphere);
        set_light_type(light_sphere);
        light_sphere.create_object_cb = create_light_sphere;
        types.add(light_sphere);
    }

    static Object * create_light_environment(AttributeAccessor& accessor, Object * parent)
    {
        Transform3f transform = GeometrySceneFile::get_transform(accessor);
        GeometryEnvironment * environment = new GeometryEnvironment(transform);

        const Vec3f intensity = accessor.get_vec3f("intensity");
        Texture * intensity_texture = dynamic_cast<Texture*>(accessor.get_object("intensity_texture"));
        TextureMultiply * intensity_multiply_texture = new TextureMultiply(intensity, intensity_texture);
        accessor.add_object("intensity_multiply_texture", intensity_multiply_texture);
        Light * light = new Light(intensity_multiply_texture);
        accessor.add_object("light", light);
        environment->set_attribute("light", light);

        GeometrySceneFile::get_opacity(accessor, environment);

        return environment;
    }

    static Object * create_light_directional(AttributeAccessor& accessor, Object * parent)
    {
        Transform3f transform = GeometrySceneFile::get_transform(accessor);
        Geometry * geometry = new Geometry(transform);

        const Vec3f intensity = accessor.get_vec3f("intensity");
        Texture * intensity_texture = dynamic_cast<Texture*>(accessor.get_object("intensity_texture"));
        TextureMultiply * intensity_multiply_texture = new TextureMultiply(intensity, intensity_texture);
        accessor.add_object("intensity_multiply_texture", intensity_multiply_texture);
        Light * light = new Light(intensity_multiply_texture);
        accessor.add_object("light", light);
        geometry->set_attribute("light", light);

        LightSamplerDirectional * light_sampler = new LightSamplerDirectional(geometry);
        accessor.add_object("light_sampler", light_sampler);
        geometry->set_attribute("light_sampler", light_sampler);

        // GeometrySceneFile::get_opacity(accessor, geometry);

        return geometry;
    }

    static Object * create_light_area(AttributeAccessor& accessor, Object * parent)
    {
        Transform3f transform = GeometrySceneFile::get_transform(accessor);
        GeometryArea * area_light = new GeometryArea(transform);

        const Vec3f intensity = accessor.get_vec3f("intensity");
        Texture * intensity_texture = dynamic_cast<Texture*>(accessor.get_object("intensity_texture"));
        TextureMultiply * intensity_multiply_texture = new TextureMultiply(intensity, intensity_texture);
        accessor.add_object("intensity_multiply_texture", intensity_multiply_texture);
        Light * light = new Light(intensity_multiply_texture);
        accessor.add_object("light", light);
        area_light->set_attribute("light", light);

        LightSamplerArea * light_sampler = new LightSamplerArea(area_light);
        accessor.add_object("light_sampler", light_sampler);
        area_light->set_attribute("light_sampler", light_sampler);

        GeometrySceneFile::get_opacity(accessor, area_light);

        // TODO: move this to some defaults.
        EmbreeGeometryArea * embree_geometry = new EmbreeGeometryArea(area_light);
        area_light->set_attribute("embree_geometry", embree_geometry);

        return area_light;
    }

    static Object * create_light_sphere(AttributeAccessor& accessor, Object * parent)
    {
        Transform3f transform = GeometrySceneFile::get_transform(accessor);
        GeometrySphere * sphere_light = new GeometrySphere(transform);

        const Vec3f intensity = accessor.get_vec3f("intensity");
        Texture * intensity_texture = dynamic_cast<Texture*>(accessor.get_object("intensity_texture"));
        TextureMultiply * intensity_multiply_texture = new TextureMultiply(intensity, intensity_texture);
        accessor.add_object("intensity_multiply_texture", intensity_multiply_texture);
        Light * light = new Light(intensity_multiply_texture);
        accessor.add_object("light", light);
        sphere_light->set_attribute("light", light);

        LightSamplerSphere * light_sampler = new LightSamplerSphere(sphere_light);
        accessor.add_object("light_sampler", light_sampler);
        sphere_light->set_attribute("light_sampler", light_sampler);

        GeometrySceneFile::get_opacity(accessor, sphere_light);

        // TODO: move this to some defaults.
        EmbreeGeometrySphere * embree_geometry = new EmbreeGeometrySphere(sphere_light);
        sphere_light->set_attribute("embree_geometry", embree_geometry);

        return sphere_light;
    }

};
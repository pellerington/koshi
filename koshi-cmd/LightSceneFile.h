#pragma once

#include "SceneFile.h"
#include "GeometrySceneFile.h"

#include <koshi/geometry/GeometryEnvironment.h>
#include <koshi/geometry/GeometryDirectional.h>

#include <koshi/light/LightSamplerArea.h>
#include <koshi/light/LightSamplerSphere.h>
#include <koshi/light/LightSamplerDirectional.h>
#include <koshi/light/LightSamplerEnvironment.h>

#include <koshi/material/MaterialLight.h>

#include <koshi/texture/TextureMultiply.h>

struct LightSceneFile
{

    static void set_light_type(Type& type)
    {
        type.reserved_attributes.push_back("intesity");
        type.reserved_attributes.push_back("intensity_texture");
        type.reserved_attributes.push_back("normalized");
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
        GeometryVisibility visibility = GeometrySceneFile::get_visibility(accessor);
        GeometryEnvironment * geometry = new GeometryEnvironment(transform, visibility);

        const Vec3f intensity = accessor.get_vec3f("intensity");
        Texture * intensity_texture = dynamic_cast<Texture*>(accessor.get_object("intensity_texture"));
        TextureMultiply * intensity_multiply_texture = new TextureMultiply(intensity, intensity_texture);
        accessor.add_object("intensity_multiply_texture", intensity_multiply_texture);

        const bool normalized = accessor.get_bool("normalized", true);

        MaterialLight * material = new MaterialLight(intensity_multiply_texture, normalized);
        accessor.add_object("material", material);
        geometry->set_attribute("material", material);

        LightSamplerEnvironment * light_sampler = new LightSamplerEnvironment(geometry);
        accessor.add_object("light_sampler", light_sampler);
        geometry->set_attribute("light_sampler", light_sampler);

        return geometry;
    }

    static Object * create_light_directional(AttributeAccessor& accessor, Object * parent)
    {
        Transform3f transform = GeometrySceneFile::get_transform(accessor);
        GeometryVisibility visibility = GeometrySceneFile::get_visibility(accessor);
        const float angle = accessor.get_float("angle");
        GeometryDirectional * geometry = new GeometryDirectional(angle, transform, visibility);

        const Vec3f intensity = accessor.get_vec3f("intensity");
        Texture * intensity_texture = dynamic_cast<Texture*>(accessor.get_object("intensity_texture"));
        TextureMultiply * intensity_multiply_texture = new TextureMultiply(intensity, intensity_texture);
        accessor.add_object("intensity_multiply_texture", intensity_multiply_texture);

        const bool normalized = accessor.get_bool("normalized", true);

        MaterialLight * material = new MaterialLight(intensity_multiply_texture, normalized);
        accessor.add_object("material", material);
        geometry->set_attribute("material", material);

        LightSamplerDirectional * light_sampler = new LightSamplerDirectional(geometry);
        accessor.add_object("light_sampler", light_sampler);
        geometry->set_attribute("light_sampler", light_sampler);

        return geometry;
    }

    static Object * create_light_area(AttributeAccessor& accessor, Object * parent)
    {
        Transform3f transform = GeometrySceneFile::get_transform(accessor);
        GeometryVisibility visibility = GeometrySceneFile::get_visibility(accessor);
        GeometryArea * geometry = new GeometryArea(transform, visibility);

        const Vec3f intensity = accessor.get_vec3f("intensity");
        Texture * intensity_texture = dynamic_cast<Texture*>(accessor.get_object("intensity_texture"));
        TextureMultiply * intensity_multiply_texture = new TextureMultiply(intensity, intensity_texture);
        accessor.add_object("intensity_multiply_texture", intensity_multiply_texture);

        const bool normalized = accessor.get_bool("normalized", true);

        MaterialLight * material = new MaterialLight(intensity_multiply_texture, normalized);
        accessor.add_object("material", material);
        geometry->set_attribute("material", material);

        LightSamplerArea * light_sampler = new LightSamplerArea(geometry);
        accessor.add_object("light_sampler", light_sampler);
        geometry->set_attribute("light_sampler", light_sampler);

        // TODO: move this to some defaults.
        EmbreeGeometryArea * embree_geometry = new EmbreeGeometryArea(geometry);
        geometry->set_attribute("embree_geometry", embree_geometry);

        return geometry;
    }

    static Object * create_light_sphere(AttributeAccessor& accessor, Object * parent)
    {
        Transform3f transform = GeometrySceneFile::get_transform(accessor);
        GeometryVisibility visibility = GeometrySceneFile::get_visibility(accessor);
        GeometrySphere * geometry = new GeometrySphere(transform, visibility);

        const Vec3f intensity = accessor.get_vec3f("intensity");
        Texture * intensity_texture = dynamic_cast<Texture*>(accessor.get_object("intensity_texture"));
        TextureMultiply * intensity_multiply_texture = new TextureMultiply(intensity, intensity_texture);
        accessor.add_object("intensity_multiply_texture", intensity_multiply_texture);

        const bool normalized = accessor.get_bool("normalized", true);

        MaterialLight * material = new MaterialLight(intensity_multiply_texture, normalized);
        accessor.add_object("material", material);
        geometry->set_attribute("material", material);

        LightSamplerSphere * light_sampler = new LightSamplerSphere(geometry);
        accessor.add_object("light_sampler", light_sampler);
        geometry->set_attribute("light_sampler", light_sampler);

        // TODO: move this to some defaults.
        EmbreeGeometrySphere * embree_geometry = new EmbreeGeometrySphere(geometry);
        geometry->set_attribute("embree_geometry", embree_geometry);

        return geometry;
    }

};
#pragma once

#include <string>
#include <fstream>

#include "../Math/Types.h"
#include "../Dependency/json.hpp"
#include "../Scene/Scene.h"
#include "../Objects/Triangle.h"
#include "../Objects/Sphere.h"
#include "../Materials/Lambert.h"
#include "../Materials/GGXReflect.h"
#include "../Materials/GGXRefract.h"
#include "../Materials/Dielectric.h"
#include "../Lights/RectangleLight.h"
#include "../Textures/Image.h"
#include "MeshFile.h"

class SceneFile
{
public:
    static Scene Import(const std::string filename)
    {
        std::ifstream input_file(filename);
        nlohmann::json scene_file;
        input_file >> scene_file;

        Scene::Settings settings;
        if(!scene_file["settings"].is_null())
        {
            settings.quality = get_float(scene_file["settings"], "quality");
            settings.max_depth = get_uint(scene_file["settings"], "max_depth");
            settings.display_lights = get_bool(scene_file["settings"], "display_lights");
            settings.sample_lights = get_bool(scene_file["settings"], "sample_lights", true);
            settings.sample_material = get_bool(scene_file["settings"], "sample_material", true);
        }

        Vec2u resolution(1);
        uint samples_per_pixel = 0;
        float focal_length = 1.f;
        Transform3f transform;
        if(!scene_file["camera"].is_null())
        {
            resolution = get_vec2u(scene_file["camera"], "resolution");
            samples_per_pixel = get_uint(scene_file["camera"], "samples_per_pixel", 1);
            focal_length = get_float(scene_file["camera"], "focal_length", 1.f);

            const float scale = get_float(scene_file["camera"], "scale", 1.f);
            const Vec3f rotation = 2.f * PI * get_vec3f(scene_file["camera"], "rotation") / 360.f;
            const Vec3f translation = get_vec3f(scene_file["camera"], "translation");
            transform = Transform3f::scale(Vec3f(scale, scale, 1.f)) * transform;
            transform = Transform3f::z_rotation(rotation[2]) * transform;
            transform = Transform3f::y_rotation(rotation[1]) * transform;
            transform = Transform3f::x_rotation(rotation[0]) * transform;
            transform = Transform3f::translation(translation) * transform;
        }
        Camera camera(transform, resolution, samples_per_pixel, focal_length);

        Scene scene(camera, settings);

        std::map<std::string, std::shared_ptr<Texture>> textures;
        if(scene_file["textures"].is_array())
        {
            for (auto it = scene_file["textures"].begin(); it != scene_file["textures"].end(); ++it)
            {
                if((*it)["type"].is_string() && (*it)["name"].is_string())
                {
                    if((*it)["type"] == "image")
                    {
                        if((*it)["filename"].is_string())
                        {
                            const bool smooth = get_bool(*it, "smooth", true);
                            std::shared_ptr<Texture> texture(new Image((*it)["filename"], smooth));
                            scene.add_texture(texture);
                            textures[(*it)["name"]] = texture;
                        }
                    }
                }
            }
        }

        std::map<std::string, std::shared_ptr<Material>> materials;
        if(scene_file["materials"].is_array())
        {
            for (auto it = scene_file["materials"].begin(); it != scene_file["materials"].end(); ++it)
            {
                if((*it)["type"].is_string() && (*it)["name"].is_string())
                {
                    if((*it)["type"] == "lambert")
                    {
                        const Vec3f diffuse_color = get_vec3f(*it, "diffuse_color");
                        const Vec3f emission = get_vec3f(*it, "emission");

                        std::shared_ptr<Material> material(new Lambert(diffuse_color, emission));
                        materials[(*it)["name"]] = material;
                        scene.add_material(material);
                    }

                    if((*it)["type"] == "ggx")
                    {
                        const Vec3f specular_color = get_vec3f(*it, "specular_color");
                        const float roughness = get_float(*it, "roughness");
                        const Vec3f emission = get_vec3f(*it, "emission");

                        std::shared_ptr<Material> material(new GGXReflect(specular_color, roughness, nullptr, emission));
                        materials[(*it)["name"]] = material;
                        scene.add_material(material);
                    }

                    if((*it)["type"] == "ggx_refract")
                    {
                        const Vec3f refractive_color = get_vec3f(*it, "refractive_color");
                        const float ior = get_float(*it, "ior", 1.f);
                        const float roughness = get_float(*it, "roughness");
                        const Vec3f emission = get_vec3f(*it, "emission");

                        std::shared_ptr<Material> material(new GGXRefract(refractive_color, roughness, ior, nullptr, emission));
                        materials[(*it)["name"]] = material;
                        scene.add_material(material);
                    }

                    if((*it)["type"] == "dielectric")
                    {
                        const Vec3f reflective_color = get_vec3f(*it, "reflective_color");
                        const Vec3f refractive_color = get_vec3f(*it, "refractive_color");
                        const float ior = get_float(*it, "ior", 1.f);
                        const float roughness = get_float(*it, "roughness");
                        const Vec3f emission = get_vec3f(*it, "emission");

                        std::shared_ptr<Material> material(new Dielectric(reflective_color, refractive_color, roughness, ior, emission));
                        materials[(*it)["name"]] = material;
                        scene.add_material(material);
                    }
                }
            }
        }

        if(scene_file["objects"].is_array())
        {
            for (auto it = scene_file["objects"].begin(); it != scene_file["objects"].end(); ++it)
            {

                if((*it)["type"] == "triangle")
                {
                    const Vec3f v0 = get_vec3f(*it, "v0");
                    const Vec3f v1 = get_vec3f(*it, "v1");
                    const Vec3f v2 = get_vec3f(*it, "v2");

                    std::shared_ptr<Material> material;
                    if((*it)["material"].is_string())
                        material = materials[(*it)["material"]];

                    std::shared_ptr<Triangle> triangle(new Triangle(v0, v1, v2, material));
                    scene.add_object(triangle);
                }

                if((*it)["type"] == "mesh")
                {
                    if((*it)["file"]["type"].is_string() && (*it)["file"]["name"].is_string())
                    {
                        std::shared_ptr<Material> material = ((*it)["material"].is_string()) ? materials[(*it)["material"]] : nullptr;
                        std::shared_ptr<Mesh> mesh;
                        if ((*it)["file"]["type"] == "obj")
                            mesh = MeshFile::ImportOBJ((*it)["file"]["name"], material);

                        if(mesh)
                            scene.add_object(mesh);
                    }
                }

                if((*it)["type"] == "sphere")
                {
                    const Vec3f position = get_vec3f(*it, "position");
                    const float scale = get_float(*it, "scale");

                    std::shared_ptr<Material> material;
                    if((*it)["material"].is_string())
                        material = materials[(*it)["material"]];

                    std::shared_ptr<Sphere> sphere(new Sphere(position, scale, material));
                    scene.add_object(sphere);
                }

            }
        }

        if(scene_file["lights"].is_array())
        {
            for (auto it = scene_file["lights"].begin(); it != scene_file["lights"].end(); ++it)
            {

                if((*it)["type"] == "rectangle")
                {
                    const Vec3f intensity = get_vec3f(*it, "intensity");
                    const Vec3f u = get_vec3f(*it, "u");
                    const Vec3f v = get_vec3f(*it, "v");
                    const Vec3f position = get_vec3f(*it, "position") - (0.5f*u + 0.5f*v);
                    const bool double_sided = get_bool(*it, "double_sided");


                    std::shared_ptr<RectangleLight> rectangle_light(new RectangleLight(position, u, v, intensity, double_sided));
                    scene.add_light(rectangle_light);
                }

                if((*it)["type"] == "environment")
                {
                    const Vec3f intensity = get_vec3f(*it, "intensity");

                    std::shared_ptr<Texture> texture;
                    if((*it)["texture"].is_string())
                        texture = textures[(*it)["texture"]];

                    std::shared_ptr<EnvironmentLight> environment_light(new EnvironmentLight(intensity, texture));
                    scene.add_light(environment_light);
                }

            }
        }

        return scene;
    }

    inline static bool get_bool(nlohmann::json &json, const std::string &name, const bool def = false)
    {
        return (json[name].is_boolean()) ? (bool)json[name] : def;
    }

    inline static float get_float(nlohmann::json &json, const std::string &name, const float def = 0.f)
    {
        return json[name].is_number() ? (float)json[name] : def;
    }

    inline static Vec3f get_vec3f(nlohmann::json &json, const std::string &name, const float def = 0.f)
    {
        Vec3f v;
        for(int i = 0; i < 3; i++)
            v[i] = json[name][i].is_number() ? (float)json[name][i] : def;
        return v;
    }

    inline static Vec2i get_vec2i(nlohmann::json &json, const std::string &name, const int def = 0)
    {
        Vec2i v;
        for(int i = 0; i < 2; i++)
            v[i] = json[name][i].is_number() ? (int)json[name][i] : def;
        return v;
    }

    inline static Vec2u get_vec2u(nlohmann::json &json, const std::string &name, const uint def = 0)
    {
        Vec2u v;
        for(int i = 0; i < 2; i++)
            v[i] = json[name][i].is_number() ? (int)json[name][i] : def;
        return v;
    }

    inline static uint get_uint(nlohmann::json &json, const std::string &name, const uint def = 0)
    {
        return json[name].is_number() ? (uint)json[name] : def;
    }
};

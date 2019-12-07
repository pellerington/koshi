#pragma once

#include <string>
#include <fstream>

#include "../Math/Types.h"
#include "../Dependency/json.hpp"
#include "../Scene/Scene.h"
#include "../Objects/ObjectSphere.h"
#include "../Objects/ObjectBox.h"
#include "../Materials/MaterialLambert.h"
#include "../Materials/MaterialBackLambert.h"
#include "../Materials/MaterialGGXReflect.h"
#include "../Materials/MaterialGGXRefract.h"
#include "../Materials/MaterialDielectric.h"
#include "../Materials/MaterialSubsurface.h"
#include "../Lights/LightArea.h"
#include "../Lights/LightSphere.h"
#include "../Lights/LightEnvironment.h"
#include "../Textures/Image.h"
#include "MeshFile.h"
#include "../Export/DebugObj.h"

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
            settings.depth = get_uint(scene_file["settings"], "depth", 2);
            settings.max_depth = get_uint(scene_file["settings"], "max_depth", 32);
            settings.sample_lights = get_bool(scene_file["settings"], "sample_lights", true);
            settings.sample_material = get_bool(scene_file["settings"], "sample_material", true);
        }

        Vec2u resolution(1);
        uint samples_per_pixel = 0;
        float focal_length = 1.f;
        Transform3f camera_transform;
        if(!scene_file["camera"].is_null())
        {
            resolution = get_vec2u(scene_file["camera"], "resolution");
            samples_per_pixel = get_uint(scene_file["camera"], "samples_per_pixel", 1);
            focal_length = get_float(scene_file["camera"], "focal_length", 1.f);

            const float scale = get_float(scene_file["camera"], "scale", 1.f);
            const Vec3f rotation = 2.f * PI * get_vec3f(scene_file["camera"], "rotation") / 360.f;
            const Vec3f translation = get_vec3f(scene_file["camera"], "translation");
            camera_transform = camera_transform * Transform3f::translation(translation);
            camera_transform = camera_transform * Transform3f::x_rotation(rotation[0]);
            camera_transform = camera_transform * Transform3f::y_rotation(rotation[1]);
            camera_transform = camera_transform * Transform3f::z_rotation(rotation[2]);
            camera_transform = camera_transform * Transform3f::scale(Vec3f(scale, scale, 1.f));
        }
        Camera camera(camera_transform, resolution, samples_per_pixel, focal_length);

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
        std::map<std::string, std::shared_ptr<Volume>> volumes;

        if(scene_file["materials"].is_array())
        {
            for (auto it = scene_file["materials"].begin(); it != scene_file["materials"].end(); ++it)
            {
                if((*it)["type"].is_string() && (*it)["name"].is_string())
                {
                    if((*it)["type"] == "lambert")
                    {
                        const Vec3f diffuse_color = get_vec3f(*it, "diffuse_color");

                        std::shared_ptr<Material> material(new MaterialLambert(diffuse_color));
                        materials[(*it)["name"]] = material;
                        scene.add_material(material);
                    }

                    if((*it)["type"] == "back_lambert")
                    {
                        const Vec3f diffuse_color = get_vec3f(*it, "diffuse_color");

                        std::shared_ptr<Material> material(new MaterialBackLambert(diffuse_color));
                        materials[(*it)["name"]] = material;
                        scene.add_material(material);
                    }

                    if((*it)["type"] == "ggx")
                    {
                        const Vec3f specular_color = get_vec3f(*it, "specular_color");
                        const float roughness = get_float(*it, "roughness");

                        std::shared_ptr<Material> material(new MaterialGGXReflect(specular_color, roughness, nullptr));
                        materials[(*it)["name"]] = material;
                        scene.add_material(material);
                    }

                    if((*it)["type"] == "ggx_refract")
                    {
                        const Vec3f refractive_color = get_vec3f(*it, "refractive_color");
                        const float ior = get_float(*it, "ior", 1.f);
                        const float roughness = get_float(*it, "roughness");

                        std::shared_ptr<Material> material(new MaterialGGXRefract(refractive_color, roughness, ior, nullptr));
                        materials[(*it)["name"]] = material;
                        scene.add_material(material);
                    }

                    if((*it)["type"] == "dielectric")
                    {
                        const Vec3f reflective_color = get_vec3f(*it, "reflective_color");
                        const Vec3f refractive_color = get_vec3f(*it, "refractive_color");
                        const float roughness = get_float(*it, "roughness");
                        const float ior = get_float(*it, "ior", 1.f);

                        std::shared_ptr<Material> material(new MaterialDielectric(reflective_color, refractive_color, roughness, ior));
                        materials[(*it)["name"]] = material;
                        scene.add_material(material);
                    }

                    if((*it)["type"] == "subsurface")
                    {
                        const float density = get_float(*it, "density");
                        const Vec3f subsurface_transparency = get_vec3f(*it, "subsurface_transparency");
                        const Vec3f subsurface_color = get_vec3f(*it, "subsurface_color");
                        const Vec3f diffuse_color = get_vec3f(*it, "diffuse_color");
                        const float diffuse_weight = get_float(*it, "diffuse_weight");

                        std::shared_ptr<Volume> volume(new Volume(density, subsurface_color, 0.f, subsurface_transparency));
                        volumes[std::string("material_") + std::string((*it)["name"])] = volume;

                        std::shared_ptr<Material> material(new MaterialSubsurface(diffuse_color, diffuse_weight));
                        materials[(*it)["name"]] = material;
                        scene.add_material(material);
                    }
                }
            }
        }

        if(scene_file["volumes"].is_array())
        {
            for (auto it = scene_file["volumes"].begin(); it != scene_file["volumes"].end(); ++it)
            {
                if((*it)["type"].is_string() && (*it)["name"].is_string())
                {
                    if((*it)["type"] == "volume")
                    {
                        const float density = get_float(*it, "density");
                        const Vec3f transparency = get_vec3f(*it, "transparency");
                        const float g = get_float(*it, "anistropy");
                        const Vec3f scattering = get_vec3f(*it, "scattering");
                        const Vec3f emission = get_vec3f(*it, "emission");

                        std::shared_ptr<Volume> volume(new Volume(density, scattering, g, transparency, emission));
                        volumes[(*it)["name"]] = volume;
                    }
                }
            }
        }

        if(scene_file["objects"].is_array())
        {
            for (auto it = scene_file["objects"].begin(); it != scene_file["objects"].end(); ++it)
            {

                if((*it)["type"] == "mesh")
                {
                    if((*it)["file"]["type"].is_string() && (*it)["file"]["name"].is_string())
                    {
                        std::shared_ptr<Material> material = ((*it)["material"].is_string()) ? materials[(*it)["material"]] : nullptr;

                        const Vec3f scale = get_vec3f(*it, "scale", 1.f);
                        const Vec3f rotation = 2.f * PI * get_vec3f(*it, "rotation") / 360.f;
                        const Vec3f translation = get_vec3f(*it, "translation");

                        Transform3f transform;
                        transform = transform * Transform3f::translation(translation);
                        transform = transform * Transform3f::z_rotation(rotation.z);
                        transform = transform * Transform3f::y_rotation(rotation.y);
                        transform = transform * Transform3f::x_rotation(rotation.x);
                        transform = transform * Transform3f::scale(scale);

                        std::shared_ptr<Volume> volume = nullptr;
                        if((*it)["material"].is_string() && volumes.find(std::string("material_") + std::string((*it)["material"])) != volumes.end())
                            volume = volumes[std::string("material_") + std::string((*it)["material"])];
                        else if((*it)["volume"].is_string())
                            volume = volumes[(*it)["volume"]];

                        std::shared_ptr<ObjectMesh> mesh;
                        if ((*it)["file"]["type"] == "obj")
                            mesh = MeshFile::ImportOBJ((*it)["file"]["name"], transform, material, volume);

                        if(mesh)
                            scene.add_object(mesh);
                    }
                }

                if((*it)["type"] == "sphere")
                {
                    const Vec3f scale = get_vec3f(*it, "scale", 1.f);
                    const Vec3f rotation = 2.f * PI * get_vec3f(*it, "rotation") / 360.f;
                    const Vec3f translation = get_vec3f(*it, "translation");

                    Transform3f transform;
                    transform = transform * Transform3f::translation(translation);
                    transform = transform * Transform3f::z_rotation(rotation.z);
                    transform = transform * Transform3f::y_rotation(rotation.y);
                    transform = transform * Transform3f::x_rotation(rotation.x);
                    transform = transform * Transform3f::scale(scale);

                    std::shared_ptr<Material> material;
                    if((*it)["material"].is_string())
                        material = materials[(*it)["material"]];

                    std::shared_ptr<Volume> volume = nullptr;
                    if((*it)["material"].is_string() && volumes.find(std::string("material_") + std::string((*it)["material"])) != volumes.end())
                        volume = volumes[std::string("material_") + std::string((*it)["material"])];
                    else if((*it)["volume"].is_string())
                        volume = volumes[(*it)["volume"]];

                    std::shared_ptr<ObjectSphere> sphere(new ObjectSphere(transform, material, volume));
                    scene.add_object(sphere);
                }

                if((*it)["type"] == "box")
                {
                    const Vec3f scale = get_vec3f(*it, "scale", 1.f);
                    const Vec3f rotation = 2.f * PI * get_vec3f(*it, "rotation") / 360.f;
                    const Vec3f translation = get_vec3f(*it, "translation");

                    Transform3f transform;
                    transform = transform * Transform3f::translation(translation);
                    transform = transform * Transform3f::z_rotation(rotation.z);
                    transform = transform * Transform3f::y_rotation(rotation.y);
                    transform = transform * Transform3f::x_rotation(rotation.x);
                    transform = transform * Transform3f::scale(scale);
                    transform = transform * Transform3f::translation(Vec3f(-0.5));

                    std::shared_ptr<Material> material;
                    if((*it)["material"].is_string())
                        material = materials[(*it)["material"]];

                    std::shared_ptr<Volume> volume = nullptr;
                    if((*it)["material"].is_string() && volumes.find(std::string("material_") + std::string((*it)["material"])) != volumes.end())
                        volume = volumes[std::string("material_") + std::string((*it)["material"])];
                    else if((*it)["volume"].is_string())
                        volume = volumes[(*it)["volume"]];

                    std::shared_ptr<ObjectBox> box(new ObjectBox(transform, material, volume));
                    scene.add_object(box);
                }
            }
        }

        if(scene_file["lights"].is_array())
        {
            for (auto it = scene_file["lights"].begin(); it != scene_file["lights"].end(); ++it)
            {

                if((*it)["type"] == "environment")
                {
                    const Vec3f intensity = get_vec3f(*it, "intensity");
                    std::shared_ptr<Light> light(new Light(intensity));

                    std::shared_ptr<Texture> texture;
                    if((*it)["texture"].is_string())
                        texture = textures[(*it)["texture"]];

                    std::shared_ptr<LightEnvironment> environment_light(new LightEnvironment(light, texture));
                    scene.add_distant_light(environment_light);
                }

                if((*it)["type"] == "area")
                {
                    const Vec3f scale = get_vec3f(*it, "scale", 1.f);
                    const Vec3f rotation = 2.f * PI * get_vec3f(*it, "rotation") / 360.f;
                    const Vec3f translation = get_vec3f(*it, "translation");

                    Transform3f transform;
                    transform = transform * Transform3f::translation(translation);
                    transform = transform * Transform3f::z_rotation(rotation.z);
                    transform = transform * Transform3f::y_rotation(rotation.y);
                    transform = transform * Transform3f::x_rotation(rotation.x);
                    transform = transform * Transform3f::scale(scale);

                    const Vec3f intensity = get_vec3f(*it, "intensity");
                    std::shared_ptr<Light> light(new Light(intensity));

                    const bool double_sided = get_bool(*it, "double_sided");
                    const bool hide_camera = get_bool(*it, "hide_camera", true);

                    std::shared_ptr<LightArea> area_light(new LightArea(transform, light, double_sided, hide_camera));
                    scene.add_light(area_light);
                }

                if((*it)["type"] == "sphere")
                {
                    const Vec3f scale = get_vec3f(*it, "scale", 1.f);
                    const Vec3f rotation = 2.f * PI * get_vec3f(*it, "rotation") / 360.f;
                    const Vec3f translation = get_vec3f(*it, "translation");

                    Transform3f transform;
                    transform = transform * Transform3f::translation(translation);
                    transform = transform * Transform3f::z_rotation(rotation.z);
                    transform = transform * Transform3f::y_rotation(rotation.y);
                    transform = transform * Transform3f::x_rotation(rotation.x);
                    transform = transform * Transform3f::scale(scale);

                    const Vec3f intensity = get_vec3f(*it, "intensity");
                    std::shared_ptr<Light> light(new Light(intensity));

                    const bool hide_camera = get_bool(*it, "hide_camera", true);

                    std::shared_ptr<LightSphere> sphere_light(new LightSphere(transform, light, hide_camera));
                    scene.add_light(sphere_light);
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

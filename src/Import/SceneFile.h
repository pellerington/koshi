#pragma once

#include <string>
#include <fstream>
#include <json.hpp>

#include <Math/Types.h>
#include <Scene/Scene.h>

#include <base/ObjectGroup.h>

#include <geometry/GeometrySphere.h>
#include <geometry/GeometryBox.h>
#include <geometry/GeometryArea.h>
#include <geometry/GeometryEnvironment.h>

#include <Materials/MaterialLambert.h>
#include <Materials/MaterialBackLambert.h>
#include <Materials/MaterialGGXReflect.h>
#include <Materials/MaterialGGXRefract.h>
#include <Materials/MaterialDielectric.h>
#include <Materials/MaterialSubsurface.h>

#include <lights/LightSamplerArea.h>
#include <lights/LightSamplerSphere.h>

#include <Textures/Image.h>
#include <Textures/Checker.h>
#include <Textures/Gradient.h>
#include <Textures/OpenVDB.h>

#include <embree/EmbreeGeometryMesh.h>
#include <embree/EmbreeGeometryArea.h>
#include <embree/EmbreeGeometrySphere.h>
#include <embree/EmbreeGeometryBox.h>

#include <integrators/IntegratorSurfaceMaterialSampler.h>
#include <integrators/IntegratorSurfaceLightSampler.h>
#include <integrators/IntegratorSurfaceMultipleImportanceSampler.h>
#include <Export/DebugObj.h>
#include <Import/MeshFile.h>

class SceneFile
{
public:
    static Scene Import(const std::string filename, Settings& settings)
    {
        std::ifstream input_file(filename);
        nlohmann::json scene_file;
        input_file >> scene_file;

        if(!scene_file["settings"].is_null())
        {
            settings.sampling_quality = get_float(scene_file["settings"], "sampling_quality");
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

        Scene scene(camera);

        std::map<std::string, Texture *> textures;
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
                            Texture * texture = new Image((*it)["filename"], smooth);
                            scene.add_object(texture);
                            textures[(*it)["name"]] = texture;
                        }
                    }

                    if((*it)["type"] == "checker")
                    {
                        const Vec3f scale = get_vec3f(*it, "scale", 0.f);
                        Texture * texture = new Checker(scale);
                        scene.add_object(texture);
                        textures[(*it)["name"]] = texture;
                    }

                    if((*it)["type"] == "gradient")
                    {
                        const Vec3f min = get_vec3f(*it, "min", 0.f);
                        const Vec3f max = get_vec3f(*it, "max", 1.f);
                        const uint axis = get_uint(*it, "axis", 0);
                        Texture * texture = new Gradient(min, max, axis);
                        scene.add_object(texture);
                        textures[(*it)["name"]] = texture;
                    }

                    if((*it)["type"] == "openvdb")
                    {
                        const std::string filename = (*it)["filename"];
                        const std::string gridname = (*it)["gridname"];

                        Texture * texture = new OpenVDB(filename, gridname, settings.num_threads);
                        scene.add_object(texture);
                        textures[(*it)["name"]] = texture;
                    }

                }
            }
        }

        std::map<std::string, Material *> materials;
        // std::map<std::string, std::shared_ptr<Volume>> volumes;

        if(scene_file["materials"].is_array())
        {
            for (auto it = scene_file["materials"].begin(); it != scene_file["materials"].end(); ++it)
            {
                if((*it)["type"].is_string() && (*it)["name"].is_string())
                {
                    if((*it)["type"] == "lambert")
                    {
                        const Vec3f diffuse_color = get_vec3f(*it, "diffuse_color");
                        Texture * diffuse_color_texture = ((*it)["diffuse_color_texture"].is_string()) ? textures[(*it)["diffuse_color_texture"]] : nullptr;

                        Material * material = new MaterialLambert(AttributeVec3f(diffuse_color_texture, diffuse_color));
                        materials[(*it)["name"]] = material;
                        scene.add_object(material);
                    }

                    if((*it)["type"] == "back_lambert")
                    {
                        const Vec3f diffuse_color = get_vec3f(*it, "diffuse_color");
                        Texture * diffuse_color_texture = ((*it)["diffuse_color_texture"].is_string()) ? textures[(*it)["diffuse_color_texture"]] : nullptr;

                        Material * material = new MaterialBackLambert(AttributeVec3f(diffuse_color_texture, diffuse_color));
                        materials[(*it)["name"]] = material;
                        scene.add_object(material);
                    }

                    if((*it)["type"] == "ggx")
                    {
                        const Vec3f specular_color = get_vec3f(*it, "specular_color");
                        Texture * specular_color_texture = ((*it)["specular_color_texture"].is_string()) ? textures[(*it)["specular_color_texture"]] : nullptr;

                        const float roughness = get_float(*it, "roughness");
                        Texture * roughness_texture = ((*it)["roughness_texture"].is_string()) ? textures[(*it)["roughness_texture"]] : nullptr;

                        Material * material = new MaterialGGXReflect(
                            AttributeVec3f(specular_color_texture, specular_color),
                            AttributeFloat(roughness_texture, roughness)
                        );
                        materials[(*it)["name"]] = material;
                        scene.add_object(material);
                    }

                    if((*it)["type"] == "ggx_refract")
                    {
                        const Vec3f refractive_color = get_vec3f(*it, "refractive_color");
                        Texture * refractive_color_texture = ((*it)["refractive_color_texture"].is_string()) ? textures[(*it)["refractive_color_texture"]] : nullptr;

                        const float roughness = get_float(*it, "roughness");
                        Texture * roughness_texture = ((*it)["roughness_texture"].is_string()) ? textures[(*it)["roughness_texture"]] : nullptr;

                        const float ior = get_float(*it, "ior", 1.f);

                        Material * material = new MaterialGGXRefract(
                            AttributeVec3f(refractive_color_texture, refractive_color),
                            AttributeFloat(roughness_texture, roughness), ior
                        );
                        materials[(*it)["name"]] = material;
                        scene.add_object(material);
                    }

                    if((*it)["type"] == "dielectric")
                    {
                        const Vec3f reflective_color = get_vec3f(*it, "reflective_color");
                        Texture * reflective_color_texture = ((*it)["reflective_color_texture"].is_string()) ? textures[(*it)["reflective_color_texture"]] : nullptr;

                        const Vec3f refractive_color = get_vec3f(*it, "refractive_color");
                        Texture * refractive_color_texture = ((*it)["refractive_color_texture"].is_string()) ? textures[(*it)["refractive_color_texture"]] : nullptr;

                        const float roughness = get_float(*it, "roughness");
                        Texture * roughness_texture = ((*it)["roughness_texture"].is_string()) ? textures[(*it)["roughness_texture"]] : nullptr;

                        const float ior = get_float(*it, "ior", 1.f);

                        Material * material = new MaterialDielectric(
                            AttributeVec3f(reflective_color_texture, reflective_color),
                            AttributeVec3f(refractive_color_texture, refractive_color),
                            AttributeFloat(roughness_texture, roughness),
                            ior
                        );
                        materials[(*it)["name"]] = material;
                        scene.add_object(material);
                    }

                    // if((*it)["type"] == "subsurface")
                    // {
                    //     const float density = get_float(*it, "density");
                    //     const Vec3f subsurface_transparency = get_vec3f(*it, "subsurface_transparency");
                    //     const Vec3f subsurface_density = Vec3f::clamp(VEC3F_ONES-subsurface_transparency, 0.f, 1.f) * density;
                    //     const Vec3f subsurface_color = get_vec3f(*it, "subsurface_color");

                    //     const Vec3f surface_color = get_vec3f(*it, "surface_color");
                    //     Texture * surface_color_texture = ((*it)["surface_color_texture"].is_string()) ? textures[(*it)["surface_color_texture"]] : nullptr;

                    //     const float surface_weight = get_float(*it, "surface_weight");
                    //     Texture * surface_weight_texture = ((*it)["surface_weight_texture"].is_string()) ? textures[(*it)["surface_weight_texture"]] : nullptr;

                    //     std::shared_ptr<Volume> volume = new Volume(subsurface_density, nullptr, subsurface_color, 0.f));
                    //     volumes[std::string("material_") + std::string((*it)["name"])] = volume;

                    //     Material * material = new MaterialSubsurface(
                    //         AttributeVec3f(surface_color_texture, surface_color),
                    //         AttributeFloat(surface_weight_texture, surface_weight)
                    //     ));
                    //     materials[(*it)["name"]] = material;
                    //     scene.add_object(material);
                    // }
                }
            }
        }

        // if(scene_file["volumes"].is_array())
        // {
        //     for (auto it = scene_file["volumes"].begin(); it != scene_file["volumes"].end(); ++it)
        //     {
        //         if((*it)["type"].is_string() && (*it)["name"].is_string())
        //         {
        //             if((*it)["type"] == "volume")
        //             {
        //                 const float density = get_float(*it, "density");
        //                 const Vec3f transparency = get_vec3f(*it, "transparency");
        //                 const Vec3f density_gain = Vec3f::clamp(VEC3F_ONES-transparency, 0.f, 1.f)*density;
        //                 Texture * density_texture = ((*it)["density_texture"].is_string()) ? textures[(*it)["density_texture"]] : nullptr;

        //                 const float g = get_float(*it, "anistropy");
        //                 const Vec3f scattering = get_vec3f(*it, "scattering");
        //                 const Vec3f emission = get_vec3f(*it, "emission");

        //                 std::shared_ptr<Volume> volume = new Volume(density_gain, density_texture, scattering, g, emission));
        //                 volumes[(*it)["name"]] = volume;
        //             }
        //         }
        //     }
        // }

        Integrator * default_integrator = new IntegratorSurfaceMultipleImportanceSampler;
        ObjectGroup * mis_integrator_group = new ObjectGroup;
        mis_integrator_group->push(new IntegratorSurfaceLightSampler);
        mis_integrator_group->push(new IntegratorSurfaceMaterialSampler);
        default_integrator->set_attribute("integrators", mis_integrator_group);

        scene.add_object(default_integrator);

        // TODO: Add user set integrators here.

        if(scene_file["objects"].is_array())
        {
            for (auto it = scene_file["objects"].begin(); it != scene_file["objects"].end(); ++it)
            {

                if((*it)["type"] == "mesh")
                {
                    if((*it)["file"]["type"].is_string() && (*it)["file"]["name"].is_string())
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

                        GeometryMesh * mesh;
                        if ((*it)["file"]["type"] == "obj")
                            mesh = MeshFile::ImportOBJ((*it)["file"]["name"], transform);
                        
                        if(!mesh) continue;

                        Material * material = ((*it)["material"].is_string()) ? materials[(*it)["material"]] : nullptr;
                        mesh->set_attribute("material", material);

                        EmbreeGeometryMesh * embree_geometry = new EmbreeGeometryMesh(mesh);
                        mesh->set_attribute("embree_geometry", embree_geometry);

                        mesh->set_attribute("integrator", default_integrator);

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

                    GeometrySphere * sphere = new GeometrySphere(transform);

                    Material * material = ((*it)["material"].is_string()) ? materials[(*it)["material"]] : nullptr;
                    sphere->set_attribute("material", material);

                    EmbreeGeometrySphere * embree_geometry = new EmbreeGeometrySphere(sphere);
                    sphere->set_attribute("embree_geometry", embree_geometry);

                    sphere->set_attribute("integrator", default_integrator);

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

                    GeometryBox * box = new GeometryBox(transform);

                    Material * material = ((*it)["material"].is_string()) ? materials[(*it)["material"]] : nullptr;
                    box->set_attribute("material", material);

                    EmbreeGeometryBox * embree_geometry = new EmbreeGeometryBox(box);
                    box->set_attribute("embree_geometry", embree_geometry);

                    box->set_attribute("integrator", default_integrator);

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
                    GeometryEnvironment * environment = new GeometryEnvironment;

                    const Vec3f intensity = get_vec3f(*it, "intensity");
                    Texture * intensity_texture = ((*it)["texture"].is_string()) ? textures[(*it)["texture"]] : nullptr;
                    // Todo: this is not destructed properly (needs to be added to the scene)
                    Light * light = new Light(AttributeVec3f(intensity_texture, intensity));

                    environment->set_attribute("light", light);

                    environment->set_attribute("integrator", default_integrator);

                    scene.add_object(environment);
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

                    GeometryArea * area = new GeometryArea(transform);

                    const Vec3f intensity = get_vec3f(*it, "intensity");
                    Texture * intensity_texture = ((*it)["intensity_texture"].is_string()) ? textures[(*it)["intensity_texture"]] : nullptr;
                    // Todo: this is not destructed properly (needs to be added to the scene)
                    Light * light = new Light(AttributeVec3f(intensity_texture, intensity));

                    area->set_attribute("light", light);

                    EmbreeGeometryArea * embree_geometry = new EmbreeGeometryArea(area);
                    area->set_attribute("embree_geometry", embree_geometry);

                    LightSamplerArea * light_sampler = new LightSamplerArea(area);
                    area->set_attribute("light_sampler", light_sampler);

                    area->set_attribute("integrator", default_integrator);

                    scene.add_object(area);
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

                    GeometrySphere * sphere = new GeometrySphere(transform);

                    const Vec3f intensity = get_vec3f(*it, "intensity");
                    Texture * intensity_texture = ((*it)["intensity_texture"].is_string()) ? textures[(*it)["intensity_texture"]] : nullptr;
                    // Todo: this is not destructed properly (needs to be added to the scene)
                    Light * light = new Light(AttributeVec3f(intensity_texture, intensity));

                    sphere->set_attribute("light", light);

                    EmbreeGeometrySphere * embree_geometry = new EmbreeGeometrySphere(sphere);
                    sphere->set_attribute("embree_geometry", embree_geometry);

                    LightSamplerSphere * light_sampler = new LightSamplerSphere(sphere);
                    sphere->set_attribute("light_sampler", light_sampler);

                    sphere->set_attribute("integrator", default_integrator);

                    scene.add_object(sphere);
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

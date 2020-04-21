#pragma once

#include <Math/Types.h>
// Scene should only relly know Object.h class
#include <geometry/Geometry.h>
#include <lights/LightSampler.h>
#include <Materials/Material.h>
#include <Textures/Texture.h>
#include <intersection/Intersect.h>
#include <Scene/Camera.h>

#include <vector>
#include <memory>
#include <map>

class Scene
{
public:
    struct Settings
    {
        uint num_threads = 1;
        uint max_depth = 2;
        uint depth = 32;
        float quality = 1;
        bool sample_lights = true;
        bool sample_material = true;
    };

    Scene() {}
    Scene(const Camera &camera, const Settings &settings) : camera(camera), settings(settings) {}

    void pre_render();

    // static void intersection_callback(const RTCFilterFunctionNArguments * args);
    // Intersect intersect(Ray &ray);

    void sample_lights(const Intersect &intersect, std::vector<LightSample> &light_samples, const float sample_multiplier, Resources &resources);

    const Camera camera;
    const Settings settings;

    bool add_object(std::shared_ptr<Geometry> object);
    bool add_material(std::shared_ptr<Material> material);
    bool add_light(std::shared_ptr<Geometry> light);
    bool add_texture(std::shared_ptr<Texture> texture);

    std::vector<std::shared_ptr<Geometry>>& get_objects() { return objects; }

private:
    std::vector<std::shared_ptr<Geometry>> objects;
    std::vector<std::shared_ptr<Material>> materials;
    std::vector<std::shared_ptr<Geometry>> lights;
    std::vector<std::shared_ptr<Texture>> textures;
};

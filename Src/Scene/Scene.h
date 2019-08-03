#pragma once

#include "../Math/Types.h"
#include "../Objects/Object.h"
#include "../Materials/Material.h"
#include "../Lights/EnvironmentLight.h"
#include "../Textures/Texture.h"

#include "Accelerator.h"
#include "Camera.h"

#include <vector>
#include <memory>

class Scene
{
public:
    struct Settings
    {
        uint max_depth = 2;
        float quality = 1;
        bool display_lights = false;
        bool sample_lights = true;
        bool sample_material = true;
    };

    Scene() {}
    Scene(const Camera &camera, const Settings &settings) : camera(camera), settings(settings) {}

    void accelerate();
    bool intersect(Ray &ray, Surface &surface);
    bool evaluate_lights(const Ray &ray, Vec3f &light, float* pdf, const LightSample* light_sample);
    inline void evaluate_light(const uint i, const Ray &ray, Vec3f &light, float* pdf);
    bool evaluate_environment_light(const Ray &ray, Vec3f &light, float* pdf = nullptr);
    bool sample_lights(const Surface &surface, std::deque<SrfSample> &srf_samples, const float sample_multiplier = 1.f);

    const Camera camera;
    const Settings settings;

    bool add_object(std::shared_ptr<Object> object);
    bool add_material(std::shared_ptr<Material> material);
    bool add_light(std::shared_ptr<Light> light);
    bool add_texture(std::shared_ptr<Texture> texture);

private:
    std::vector<std::shared_ptr<Object>> objects;
    std::vector<std::shared_ptr<Material>> materials;
    std::vector<std::shared_ptr<Light>> lights;
    std::vector<std::shared_ptr<Texture>> textures;
    std::shared_ptr<Light> environment_light;

    std::unique_ptr<Accelerator> accelerator;
};

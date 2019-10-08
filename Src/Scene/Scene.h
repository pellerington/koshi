#pragma once

#include "../Math/Types.h"
#include "../Objects/Object.h"
#include "../Materials/Material.h"
#include "../Lights/LightEnvironment.h"
#include "../Textures/Texture.h"
#include "../Volume/VolumeStack.h"

#include "Embree.h"
#include "Camera.h"

#include <vector>
#include <memory>
#include <map>

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

    void pre_render();
    static void get_volumes_callback(const RTCFilterFunctionNArguments * args);
    Surface intersect(Ray &ray, VolumeStack * volume_stack);
    bool evaluate_lights(const Ray &ray, std::deque<LightSample> &light_results);
    Vec3f evaluate_environment_light(const Ray &ray);
    bool sample_lights(const Surface &surface, std::deque<LightSample> &light_samples, const float sample_multiplier = 1.f);

    const Camera camera;
    const Settings settings;

    bool add_object(std::shared_ptr<Object> object);
    bool add_material(std::shared_ptr<Material> material);
    bool add_light(std::shared_ptr<Light> light);
    bool add_texture(std::shared_ptr<Texture> texture);

private:

    std::map<uint, std::shared_ptr<Object>> rtc_to_obj;
    RTCScene rtc_scene;

    std::vector<std::shared_ptr<Object>> objects;
    std::vector<std::shared_ptr<Material>> materials;
    std::vector<std::shared_ptr<Light>> lights;
    std::vector<std::shared_ptr<Texture>> textures;
    std::shared_ptr<Light> environment_light;
};

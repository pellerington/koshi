#pragma once

#include "../Math/Types.h"
#include "../Objects/Object.h"
#include "../Lights/LightCombiner.h"
#include "../Materials/Material.h"
#include "../Textures/Texture.h"
#include "../Util/Intersect.h"
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
        uint depth = 32;
        float quality = 1;
        bool sample_lights = true;
        bool sample_material = true;
    };

    Scene() {}
    Scene(const Camera &camera, const Settings &settings) : camera(camera), settings(settings) {}

    void pre_render();

    struct IntersectContext : public RTCIntersectContext {
        Scene * scene; // Make this one const
        Ray * ray;
        VolumeStack * volumes;
    };
    static void intersection_callback(const RTCFilterFunctionNArguments * args);
    Intersect intersect(Ray &ray);

    void sample_lights(const Surface &surface, std::vector<LightSample> &light_samples, RNG &rng, const float sample_multiplier);

    const Camera camera;
    const Settings settings;

    bool add_object(std::shared_ptr<Object> object);
    bool add_material(std::shared_ptr<Material> material);
    bool add_distant_light(std::shared_ptr<Object> light);
    bool add_light(std::shared_ptr<Object> light);
    bool add_texture(std::shared_ptr<Texture> texture);

private:
    std::map<uint, std::shared_ptr<Object>> rtc_to_obj;
    RTCScene rtc_scene;

    std::vector<std::shared_ptr<Object>> objects;
    std::vector<std::shared_ptr<Material>> materials;
    std::vector<std::shared_ptr<Object>> lights;
    std::shared_ptr<LightCombiner> distant_lights = std::shared_ptr<LightCombiner>(new LightCombiner);
    std::vector<std::shared_ptr<Texture>> textures;
};

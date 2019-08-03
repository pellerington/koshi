#pragma once

#include "Light.h"
#include "../Textures/Texture.h"

class EnvironmentLight : public Light
{
public:
    EnvironmentLight(Vec3f intensity, std::shared_ptr<Texture> texture);
    bool evaluate_light(const Ray &ray, Vec3f &light, float* pdf = nullptr);
    bool sample_light(const uint num_samples, const Surface &surface, std::deque<LightSample> &light_samples);
    const uint estimated_samples(const Surface &surface) { return 0; }
private:
    std::shared_ptr<Texture> texture;
};

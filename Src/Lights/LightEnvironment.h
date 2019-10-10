#pragma once

#include "Light.h"
#include "../Textures/Texture.h"

class LightEnvironment : public Light
{
public:
    LightEnvironment(const Vec3f &intensity, std::shared_ptr<Texture> texture);
    bool evaluate_light(const Ray &ray, LightSample &light_samples);
    bool sample_light(const uint num_samples, const Surface &surface, std::vector<LightSample> &light_samples);
    const uint estimated_samples(const Surface &surface) { return 0; }
private:
    const std::shared_ptr<Texture> texture;
};

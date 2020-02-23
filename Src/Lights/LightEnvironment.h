#pragma once

#include "../Textures/Texture.h"
#include "../Objects/Object.h"
#include "Light.h"

class LightEnvironment : public Object
{
public:
    LightEnvironment(std::shared_ptr<Light> light = nullptr, std::shared_ptr<Texture> texture = nullptr);

    Type get_type() { return Object::LightEnvironment; }

    bool sample_light(const uint num_samples, const Vec3f * pos, const Vec3f * pfar, std::vector<LightSample> &light_samples, Resources &resources);
    bool evaluate_light(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample, Resources &resources);
private:
    const std::shared_ptr<Texture> texture;
};

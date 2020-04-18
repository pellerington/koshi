#pragma once

#include <Textures/Texture.h>
#include <geometry/Geometry.h>
#include <Lights/Light.h>

class LightEnvironment : public Geometry
{
public:
    LightEnvironment(std::shared_ptr<Light> light = nullptr, std::shared_ptr<Texture> texture = nullptr);

    bool sample_light(const uint num_samples, const Vec3f * pos, const Vec3f * pfar, std::vector<LightSample> &light_samples, Resources &resources);
    bool evaluate_light(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample, Resources &resources);
private:
    const std::shared_ptr<Texture> texture;
};

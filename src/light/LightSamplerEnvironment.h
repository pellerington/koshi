#pragma once

#include <koshi/light/LightSampler.h>
#include <koshi/geometry/GeometryEnvironment.h>

class LightSamplerEnvironment : public LightSampler
{
public:
    LightSamplerEnvironment(GeometryEnvironment * geometry) : geometry(geometry)
    {
    }

    void pre_render(Resources& resources);

    // TODO: Create an instance with sample evaluate in it like materials?
    struct LightSamplerDataEnvironment : public LightSamplerData
    {
        Random<2> rng;
    };
    const LightSamplerData * pre_integrate(const Surface * surface, Resources& resources) const;

    bool sample(LightSample& sample, const LightSamplerData * data, Resources& resources) const;
    float evaluate(const Intersect * intersect, const LightSamplerData * data, Resources& resources) const;

    LightType get_light_type() const { return LightType::AREA; }

private:
    GeometryEnvironment * geometry;
    Material * material;

    // TODO: Move this into a texture sampler?
    std::vector<float> cdfv;
    std::vector<std::vector<float>> cdfu;
};
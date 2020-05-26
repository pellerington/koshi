#pragma once

#include <lights/LightSampler.h>
#include <geometry/GeometrySphere.h>

class LightSamplerSphere : public LightSampler
{
public:
    LightSamplerSphere(GeometrySphere * geometry);

    bool sample_light(const uint num_samples, const GeometrySurface * surface, std::vector<LightSample>& light_samples, Resources& resources);
    float evaluate_light(const Intersect * intersect, const GeometrySurface * surface, Resources& resources);

    bool sample_sa(const uint num_samples, const GeometrySurface * surface, std::vector<LightSample>& light_samples, Resources& resources);
    float evaluate_sa(const Intersect * intersect, const GeometrySurface * surface, Resources& resources);

    bool sample_area(const uint num_samples, const GeometrySurface * surface, std::vector<LightSample>& light_samples, Resources& resources);
    float evaluate_area(const Intersect * intersect, const GeometrySurface * surface, Resources& resources);

private:
    GeometrySphere * geometry;
    Light * light;
    
    float area;
};

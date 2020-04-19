#pragma once

#include <lights/LightSampler.h>
#include <geometry/GeometrySphere.h>

class LightSamplerSphere : public LightSampler
{
public:
    LightSamplerSphere(GeometrySphere * geometry);

    bool sample_light(const uint num_samples, const Vec3f * pos, const Vec3f * pfar, std::vector<LightSample> &light_samples, Resources &resources);
    bool evaluate_light(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample, Resources &resources);

    bool sample_sa(const uint num_samples, const Vec3f * pos, const Vec3f * pfar, std::vector<LightSample> &light_samples, Resources &resources);
    bool evaluate_sa(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample, Resources &resources);

    bool sample_area(const uint num_samples, const Vec3f * pos, const Vec3f * pfar, std::vector<LightSample> &light_samples, Resources &resources);
    bool evaluate_area(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample, Resources &resources);

private:
    GeometrySphere * geometry;
    Light * light;
    
    float area;
};

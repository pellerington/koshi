#pragma once

#include "../Objects/ObjectSphere.h"

class LightSphere : public ObjectSphere
{
public:
    LightSphere(const Transform3f &obj_to_world = Transform3f(), std::shared_ptr<Light> light = nullptr, const bool hide_camera = true);

    Type get_type() { return Object::LightSphere; }

    bool sample_light(const uint num_samples, const Vec3f * pos, const Vec3f * pfar, std::vector<LightSample> &light_samples, RNG &rng);
    bool evaluate_light(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample);

    bool sample_sa(const uint num_samples, const Vec3f * pos, const Vec3f * pfar, std::vector<LightSample> &light_samples, RNG &rng);
    bool evaluate_sa(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample);

    bool sample_area(const uint num_samples, const Vec3f * pos, const Vec3f * pfar, std::vector<LightSample> &light_samples, RNG &rng);
    bool evaluate_area(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample);

private:
    float area;
};

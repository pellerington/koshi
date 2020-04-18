#pragma once

#include  <geometry/Geometry.h>

class LightArea : public Geometry
{
public:
    LightArea(const Transform3f &obj_to_world = Transform3f(), std::shared_ptr<Light> light = nullptr, const bool double_sided = false, const bool hide_camera = true);

    bool sample_light(const uint num_samples, const Vec3f * pos, const Vec3f * pfar, std::vector<LightSample> &light_samples, Resources &resources);
    bool evaluate_light(const Surface &intersect, const Vec3f * pos, const Vec3f * pfar, LightSample &light_sample, Resources &resources);
private:
    bool double_sided;
    Vec3f normal;
    float area;
};

#pragma once

#include <lights/LightSampler.h>
#include <geometry/Geometry.h>

// TODO: Rename "Directional" light so we dont confuse with SurfaceDistant type.
// TODO: Add a GeometryDirectional class.
class LightSamplerDirectional : public LightSampler
{
public:
    LightSamplerDirectional(Geometry * geometry);

    bool sample_light(const uint num_samples, const Surface * surface, std::vector<LightSample>& light_samples, Resources& resources);

    float evaluate_light(const Intersect * intersect, const Surface * surface, Resources &resources)
    {
        return 0.f;
    }

private:
    Geometry * geometry;
    Light * light;

    Vec3f direction;
};

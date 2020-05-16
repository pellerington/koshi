#pragma once

#include <lights/LightSampler.h>
#include <geometry/Geometry.h>

// TODO: Add a GeometryDistant class.
class LightSamplerDistant : public LightSampler
{
public:
    LightSamplerDistant(Geometry * geometry);

    bool sample_light(const uint num_samples, const Intersect * intersect, std::vector<LightSample>& light_samples, Resources& resources);

    float evaluate_light(const Intersect * light_intersect, const Intersect * intersect, Resources &resources)
    {
        return 0.f;
    }

private:
    // TODO: Add it's own geometry distant.
    Geometry * geometry;
    Light * light;

    Vec3f direction;
};

#include "Lambert.h"

#include "../Math/RNG.h"
#include "../Math/Helpers.h"
#include <cmath>
#include <iostream>

Lambert::Lambert(const Vec3f &diffuse_color, const Vec3f &emission)
: diffuse_color(diffuse_color), emission(emission)
{
}

std::shared_ptr<Material> Lambert::instance(const Surface &surface)
{
    std::shared_ptr<Lambert> material(new Lambert(*this));
    return material;
}

bool Lambert::sample_material(const Surface &surface, std::deque<MaterialSample> &samples, const float sample_reduction)
{
    if(!surface.enter)
        return false;

    const uint num_samples = std::max(1.f, SAMPLES_PER_SA * sample_reduction);
    const float quality = 1.f / num_samples;

    const Transform3f transform = Transform3f::normal_transform(surface.normal);

    std::vector<Vec2f> rnd;
    RNG::Rand2d(num_samples, rnd);

    for(uint i = 0; i < rnd.size(); i++)
    {
        samples.emplace_back();
        MaterialSample &sample = samples.back();
        sample.quality = quality;

#if UNIFORM_SAMPLE
        // Uniform Sample
        const float theta = TWO_PI * rnd[i][0];
        const float phi = acosf(rnd[i][1]);
        sample.wo = transform * Vec3f(sinf(phi) * cosf(theta), cosf(phi), sinf(phi) * sinf(theta));

        sample.fr = diffuse_color * INV_PI * sample.wo.dot(surface.normal);;
        sample.pdf = INV_TWO_PI;
#else
        // Cosine Sample
        const float theta = TWO_PI * rnd[i][0];
        const float r = sqrtf(rnd[i][1]);
        const float x = r * cosf(theta), z = r * sinf(theta), y = sqrtf(std::max(EPSILON_F, 1.f - rnd[i][1]));
        sample.wo = transform * Vec3f(x, y, z);

        sample.fr = diffuse_color * INV_PI * sample.wo.dot(surface.normal);
        sample.pdf = y * INV_PI;
#endif
    }


    return true;
}

bool Lambert::evaluate_material(const Surface &surface, MaterialSample &sample)
{
    if(sample.wo.dot(surface.normal) < 0)
        return false;

    sample.fr = diffuse_color * INV_PI * sample.wo.dot(surface.normal);

#if UNIFORM_SAMPLE
    sample.pdf = INV_TWO_PI;
#else
    sample.pdf = surface.normal.dot(sample.wo) * INV_PI;
#endif

    return true;
}

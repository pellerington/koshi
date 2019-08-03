#include "Lambert.h"

#include "../Math/RNG.h"
#include "../Math/Helpers.h"
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>

#define UNIFORM_SAMPLE 0

Lambert::Lambert(Vec3f diffuse_color, Vec3f emission)
: diffuse_color(diffuse_color), emission(emission)
{
}

bool Lambert::sample_material(const Surface &surface, std::deque<SrfSample> &srf_samples, float sample_reduction)
{
    const uint num_samples = std::max(1.f, SAMPLES_PER_SA * sample_reduction);

    Eigen::Matrix3f transform = world_transform(surface.normal);

    std::vector<Vec2f> rnd;
    RNG::Rand2d(num_samples, rnd);

    for(uint i = 0; i < rnd.size(); i++)
    {
        srf_samples.emplace_back();
        SrfSample &srf_sample = srf_samples.back();

#if UNIFORM_SAMPLE
        // Uniform Sample
        float theta = TWO_PI * rnd[i][0];
        float phi = acosf(rnd[i][1]);
        srf_sample.wo = transform * Vec3f(sinf(phi) * cosf(theta), cosf(phi), sinf(phi) * sinf(theta));
        srf_sample.fr = diffuse_color * INV_PI;
        srf_sample.pdf = INV_TWO_PI;
#else
        // Cosine Sample
        float theta = TWO_PI * rnd[i][0];
        float r = sqrtf(rnd[i][1]);
        float x = r * cosf(theta), z = r * sinf(theta), y = sqrtf(std::max(EPSILON, 1.f - rnd[i][1]));
        srf_sample.wo = transform * Vec3f(x, y, z);
        srf_sample.fr = diffuse_color * INV_PI;
        srf_sample.pdf = y * INV_PI;
#endif
    }


    return true;
}

bool Lambert::evaluate_material(const Surface &surface, SrfSample &srf_sample, float &pdf)
{
    if(srf_sample.wo.dot(surface.normal) < 0)
        return false;

    srf_sample.fr = diffuse_color * INV_PI;

#if UNIFORM_SAMPLE
    pdf = INV_TWO_PI;
#else
    pdf = surface.normal.dot(srf_sample.wo) * INV_PI;
#endif

    return true;
}

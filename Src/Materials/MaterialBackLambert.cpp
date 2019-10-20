#include "MaterialBackLambert.h"

#include "../Math/RNG.h"
#include "../Math/Helpers.h"
#include <cmath>
#include <iostream>

MaterialBackLambert::MaterialBackLambert(const Vec3f &diffuse_color, const Vec3f &emission)
: diffuse_color(diffuse_color), emission(emission)
{
}

std::shared_ptr<Material> MaterialBackLambert::instance(const Surface * surface)
{
    std::shared_ptr<MaterialBackLambert> material(new MaterialBackLambert(*this));
    material->surface = surface;
    return material;
}

bool MaterialBackLambert::sample_material(std::vector<MaterialSample> &samples, const float sample_reduction)
{
    if(!surface)
        return false;

    const uint num_samples = std::max(1.f, SAMPLES_PER_SA * sample_reduction);
    const float quality = 1.f / num_samples;

    std::vector<Vec2f> rnd;
    RNG::Rand2d(num_samples, rnd);

    for(uint i = 0; i < rnd.size(); i++)
    {
        samples.emplace_back();
        MaterialSample &sample = samples.back();
        sample.quality = quality;

        const float theta = TWO_PI * rnd[i][0];
        const float r = sqrtf(rnd[i][1]);
        const float x = r * cosf(theta), z = r * sinf(theta), y = ((surface->enter) ? -1.f : 1.f) * sqrtf(std::max(EPSILON_F, 1.f - rnd[i][1]));
        sample.wo = surface->transform * Vec3f(x, y, z);


        sample.fr = diffuse_color * INV_PI * fabs(sample.wo.dot(surface->normal));
        sample.pdf = fabs(y) * INV_PI;        
    }

    return true;
}

bool MaterialBackLambert::evaluate_material(MaterialSample &sample)
{

    if(!surface || sample.wo.dot(surface->normal) * surface->n_dot_wi > 0.f)
        return false;

    const float n_dot_wo = fabs(sample.wo.dot(surface->normal));
    sample.fr = diffuse_color * INV_PI * n_dot_wo;
    sample.pdf = n_dot_wo * INV_PI;

    return true;
}

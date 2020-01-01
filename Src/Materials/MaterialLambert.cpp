#include "MaterialLambert.h"

#include "../Math/Helpers.h"
#include "../Util/Color.h"
#include <cmath>
#include <iostream>

MaterialLambert::MaterialLambert(const AttributeVec3f &diffuse_color_attr)
: diffuse_color_attr(diffuse_color_attr)
{
}

std::shared_ptr<Material> MaterialLambert::instance(const Surface * surface, RNG &rng)
{
    std::shared_ptr<MaterialLambert> material(new MaterialLambert(*this));
    material->surface = surface;
    material->rng = &rng;
    material->diffuse_color = diffuse_color_attr.get_value(surface->u, surface->v, 0.f);
    return material;
}

bool MaterialLambert::sample_material(std::vector<MaterialSample> &samples, const float sample_reduction)
{
    if(!surface || !surface->front)
        return false;

    const uint num_samples = std::max(1.f, SAMPLES_PER_SA * sample_reduction);
    const float quality = 1.f / SAMPLES_PER_SA;
    rng->Reset2D();

    for(uint i = 0; i < num_samples; i++)
    {
        const Vec2f rnd = rng->Rand2D();

        samples.emplace_back();
        MaterialSample &sample = samples.back();
        sample.quality = quality;

#if UNIFORM_SAMPLE
        // Uniform Sample
        const float theta = TWO_PI * rnd[0];
        const float phi = acosf(rnd[1]);
        sample.wo = surface->transform * Vec3f(sinf(phi) * cosf(theta), cosf(phi), sinf(phi) * sinf(theta));
        sample.pdf = INV_TWO_PI;
#else
        // Cosine Sample
        const float theta = TWO_PI * rnd[0];
        const float r = sqrtf(rnd[1]);
        const float x = r * cosf(theta), z = r * sinf(theta), y = sqrtf(std::max(EPSILON_F, 1.f - rnd[1]));
        sample.wo = surface->transform * Vec3f(x, y, z);
        sample.pdf = y * INV_PI;
#endif
        sample.weight = diffuse_color * INV_PI * sample.wo.dot(surface->normal);
        sample.type = MaterialSample::Diffuse;
    }

    return true;
}

bool MaterialLambert::evaluate_material(MaterialSample &sample)
{
    if(!surface || !surface->front)
        return false;

    const float n_dot_wo = surface->normal.dot(sample.wo);
    if(n_dot_wo < 0.f)
        return 0.f;

    sample.weight = diffuse_color * INV_PI * n_dot_wo;

#if UNIFORM_SAMPLE
    sample.pdf = INV_TWO_PI;
#else
    sample.pdf = n_dot_wo * INV_PI;
#endif

    return true;
}

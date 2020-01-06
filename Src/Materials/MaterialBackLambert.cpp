#include "MaterialBackLambert.h"

#include "../Math/Helpers.h"
#include <cmath>
#include <iostream>

MaterialBackLambert::MaterialBackLambert(const AttributeVec3f &diffuse_color_attr)
: diffuse_color_attr(diffuse_color_attr)
{
}

std::shared_ptr<MaterialInstance> MaterialBackLambert::instance(const Surface * surface)
{
    std::shared_ptr<MaterialInstanceBackLambert> instance(new MaterialInstanceBackLambert);
    instance->surface = surface;
    instance->diffuse_color = diffuse_color_attr.get_value(surface->u, surface->v, 0.f);
    return instance;
}

bool MaterialBackLambert::sample_material(const MaterialInstance * material_instance, std::vector<MaterialSample> &samples, RNG &rng, const float sample_reduction)
{
    const MaterialInstanceBackLambert * instance = dynamic_cast<const MaterialInstanceBackLambert *>(material_instance);

    const uint num_samples = std::max(1.f, SAMPLES_PER_SA * sample_reduction);
    const float quality = 1.f / SAMPLES_PER_SA;
    rng.Reset2D();

    for(uint i = 0; i < num_samples; i++)
    {
        const Vec2f rnd = rng.Rand2D();

        samples.emplace_back();
        MaterialSample &sample = samples.back();
        sample.quality = quality;

        const float theta = TWO_PI * rnd[0];
        const float r = sqrtf(rnd[1]);
        const float x = r * cosf(theta), z = r * sinf(theta), y = ((instance->surface->front) ? -1.f : 1.f) * sqrtf(std::max(EPSILON_F, 1.f - rnd[1]));
        sample.wo = instance->surface->transform * Vec3f(x, y, z);
        sample.weight = instance->diffuse_color * INV_PI * fabs(sample.wo.dot(instance->surface->normal));
        sample.pdf = fabs(y) * INV_PI;
        sample.type = MaterialSample::Diffuse;
    }

    return true;
}

bool MaterialBackLambert::evaluate_material(const MaterialInstance * material_instance, MaterialSample &sample)
{
    const MaterialInstanceBackLambert * instance = dynamic_cast<const MaterialInstanceBackLambert *>(material_instance);

    if(sample.wo.dot(instance->surface->normal) * instance->surface->n_dot_wi > 0.f)
        return false;

    const float n_dot_wo = fabs(sample.wo.dot(instance->surface->normal));
    sample.weight = instance->diffuse_color * INV_PI * n_dot_wo;
    sample.pdf = n_dot_wo * INV_PI;

    return true;
}

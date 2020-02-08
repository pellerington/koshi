#include "MaterialGGXRefract.h"

#include "../Math/Helpers.h"
#include "../Util/Color.h"
#include <cmath>
#include <iostream>

MaterialGGXRefract::MaterialGGXRefract(const AttributeVec3f &refractive_color_attribute, const AttributeFloat &roughness_attribute, const float &ior)
: refractive_color_attribute(refractive_color_attribute), roughness_attribute(roughness_attribute), ior(ior)
{
}

MaterialInstance * MaterialGGXRefract::instance(const Surface * surface, Resources &resources)
{
    MaterialInstanceGGXRefract * instance = resources.memory.create<MaterialInstanceGGXRefract>();
    instance->surface = surface;
    instance->ior_in = surface->curr_ior;
    instance->ior_out = surface->front ? ior : surface->prev_ior;
    instance->refractive_color = refractive_color_attribute.get_value(surface->u, surface->v, 0.f);
    instance->roughness = roughness_attribute.get_value(surface->u, surface->v, 0.f);
    instance->roughness = clamp(instance->roughness * instance->roughness, 0.01f, 0.99f);
    instance->roughness_sqr = instance->roughness * instance->roughness;
    instance->fresnel = resources.memory.create<FresnelDielectric>(instance->ior_in, instance->ior_out);
    return instance;
}

bool MaterialGGXRefract::sample_material(const MaterialInstance * material_instance, std::vector<MaterialSample> &samples, const float sample_reduction, Resources &resources)
{
    const MaterialInstanceGGXRefract * instance = (const MaterialInstanceGGXRefract *)material_instance;

    // Estimate the number of samples
    uint num_samples = SAMPLES_PER_SA * sqrtf(instance->roughness);
    const float quality = 1.f / num_samples;
    num_samples = std::max(1.f, num_samples * sample_reduction);
    RNG &rng = resources.rng; rng.Reset2D();

    for(uint i = 0; i < num_samples; i++)
    {
        const Vec2f rnd = rng.Rand2D();

        const float theta = TWO_PI * rnd[0];
        const float phi = atanf(instance->roughness * sqrtf(rnd[1]) / sqrtf(1.f - rnd[1]));
        const Vec3f h = ((instance->surface->front) ? 1.f : -1.f) * (instance->surface->transform * Vec3f(sinf(phi) * cosf(theta), cosf(phi), sinf(phi) * sinf(theta)));

        const float h_dot_wi = clamp(h.dot(-instance->surface->wi), -1.f, 1.f);
        const float eta = instance->ior_in / instance->ior_out;
        const float k = 1.f - eta * eta * (1.f - h_dot_wi * h_dot_wi);
        if(k < 0.f) continue;

        samples.emplace_back();
        MaterialSample &sample = samples.back();
        sample.quality = quality;
        sample.wo = eta * instance->surface->wi + (eta * fabs(h_dot_wi) - sqrtf(k)) * h;

        const Vec3f normal = ((instance->surface->front) ? instance->surface->normal : -instance->surface->normal);
        const float n_dot_h = clamp(h.dot(normal), -1.f, 1.f);
        const float h_dot_wo = clamp(h.dot(sample.wo), -1.f, 1.f);
        const float n_dot_wi = clamp(normal.dot(-instance->surface->wi), -1.f, 1.f);
        const float n_dot_wo = clamp(normal.dot(sample.wo), -1.f, 1.f);

        const float d = D(normal, h, n_dot_h, instance->roughness_sqr);
        const float g = G1(-instance->surface->wi, normal, h, h_dot_wi, n_dot_wi, instance->roughness_sqr)*G1(sample.wo, normal, h, h_dot_wo, n_dot_wo, instance->roughness_sqr);
        const Vec3f f = instance->fresnel->Ft(fabs(h_dot_wi));

        const float denom = std::pow(instance->ior_in * h_dot_wi + instance->ior_out * h_dot_wo + EPSILON_F, 2);

        sample.weight = instance->ior_out * instance->ior_out * (fabs(h_dot_wi) * fabs(h_dot_wo)) / (fabs(n_dot_wi) * fabs(n_dot_wo));
        sample.weight *= instance->refractive_color * f * g * d * fabs(n_dot_wo) / denom;
        // sample.weight *= eta * eta;

        sample.pdf = d * n_dot_h * (instance->ior_out * instance->ior_out * fabs(h_dot_wo)) / denom;
        sample.type = (instance->roughness < 0.02f) ? MaterialSample::Specular : MaterialSample::Glossy;

        if(!sample.pdf)
            samples.pop_back();
    }

    return true;
}

bool MaterialGGXRefract::evaluate_material(const MaterialInstance * material_instance, MaterialSample &sample)
{
    // Currently see little reason to evalualte this. Prehaps on the condition that we are exiting the object.
    return false;

    const MaterialInstanceGGXRefract * instance = (const MaterialInstanceGGXRefract *)material_instance;

    if(sample.wo.dot(instance->surface->wi) > 0.f)
        return false;

    const Vec3f normal = ((instance->surface->front) ? instance->surface->normal : -instance->surface->normal);
    const Vec3f h = ((instance->surface->front) ? 1.f : -1.f) * (-instance->surface->wi*instance->ior_in + sample.wo*instance->ior_out).normalized();

    const float n_dot_h = clamp(h.dot(normal), -1.f, 1.f);
    const float h_dot_wi = clamp(h.dot(-instance->surface->wi), -1.f, 1.f);
    const float h_dot_wo = clamp(h.dot(sample.wo), -1.f, 1.f);
    const float n_dot_wi = clamp(normal.dot(-instance->surface->wi), -1.f, 1.f);
    const float n_dot_wo = clamp(normal.dot(sample.wo), -1.f, 1.f);

    const float d = D(normal, h, n_dot_h, instance->roughness_sqr);
    const float g = G1(-instance->surface->wi, normal, h, h_dot_wi, n_dot_wi, instance->roughness_sqr)*G1(sample.wo, normal, h, h_dot_wo, n_dot_wo, instance->roughness_sqr);
    const Vec3f f = instance->fresnel->Fr(fabs(h_dot_wi));

    const float denom = std::pow(instance->ior_in * h_dot_wi + instance->ior_out * h_dot_wo + EPSILON_F, 2);

    sample.weight = instance->ior_out * instance->ior_out * (fabs(h_dot_wi) * fabs(h_dot_wo)) / (fabs(n_dot_wi) * fabs(n_dot_wo));
    sample.weight *= instance->refractive_color * f * g * d * fabs(n_dot_wo) / denom;
    // sample.weight *= eta * eta;

    sample.pdf = d * n_dot_h * (instance->ior_out * instance->ior_out * fabs(h_dot_wo)) / denom;

    return true;
}

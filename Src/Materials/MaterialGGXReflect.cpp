#include "MaterialGGXReflect.h"

#include "../Math/Helpers.h"
#include "../Util/Color.h"
#include <cmath>
#include <iostream>

MaterialGGXReflect::MaterialGGXReflect(const AttributeVec3f &specular_color_attribute, const AttributeFloat &roughness_attribute)
: specular_color_attribute(specular_color_attribute), roughness_attribute(roughness_attribute)
{
}

std::shared_ptr<MaterialInstance> MaterialGGXReflect::instance(const Surface * surface)
{
    std::shared_ptr<MaterialInstanceGGXReflect> instance(new MaterialInstanceGGXReflect);
    instance->surface = surface;

    instance->specular_color = specular_color_attribute.get_value(surface->u, surface->v, 0.f);

    instance->roughness = roughness_attribute.get_value(surface->u, surface->v, 0.f);
    instance->roughness = clamp(instance->roughness * instance->roughness, 0.01f, 0.99f);
    instance->roughness_sqr = instance->roughness * instance->roughness;

    instance->fresnel = std::shared_ptr<Fresnel>(new FresnelMetalic(instance->specular_color));

    return instance;
}

bool MaterialGGXReflect::sample_material(const MaterialInstance * material_instance, std::vector<MaterialSample> &samples, RNG &rng, const float sample_reduction)
{
    const MaterialInstanceGGXReflect * instance = dynamic_cast<const MaterialInstanceGGXReflect *>(material_instance);

    // Estimate the number of samples
    uint num_samples = SAMPLES_PER_SA * sqrtf(instance->roughness);
    const float quality = 1.f / num_samples;
    num_samples = std::max(1.f, num_samples * sample_reduction);
    rng.Reset2D();

    for(uint i = 0; i < num_samples; i++)
    {
        const Vec2f rnd = rng.Rand2D();

        const float theta = TWO_PI * rnd[0];
        const float phi = atanf(instance->roughness * sqrtf(rnd[1]) / sqrtf(1.f - rnd[1]));
        const Vec3f h = (instance->surface->front ? 1.f : -1.f) * (instance->surface->transform * Vec3f(sinf(phi) * cosf(theta), cosf(phi), sinf(phi) * sinf(theta)));
        const float h_dot_wi = clamp(h.dot(-instance->surface->wi), -1.f, 1.f);

        // If we are inside the only time we want to call this is if we have total internal reflection.
        const Vec3f f = instance->fresnel->Fr(fabs(h_dot_wi));
        if(!instance->surface->front && f < 1.f)
            continue;

        samples.emplace_back();
        MaterialSample &sample = samples.back();
        sample.quality = quality;
        sample.wo = (2.f * h_dot_wi * h + instance->surface->wi);

        const Vec3f normal = (instance->surface->front ? 1.f : -1.f) * instance->surface->normal;
        const float n_dot_h = clamp(h.dot(normal), -1.f, 1.f);
        const float h_dot_wo = clamp(h.dot(sample.wo), -1.f, 1.f);
        const float n_dot_wi = fabs(instance->surface->n_dot_wi);
        const float n_dot_wo = clamp(normal.dot(sample.wo), -1.f, 1.f);

        const float d = D(normal, h, n_dot_h, instance->roughness_sqr);
        const float g = G1(-instance->surface->wi, normal, h, h_dot_wi, n_dot_wi, instance->roughness_sqr)*G1(sample.wo, normal, h, h_dot_wo, n_dot_wo, instance->roughness_sqr);

        sample.weight = (n_dot_wo > 0.f) ? (instance->specular_color * f * g * d) / (4.f * n_dot_wi) : VEC3F_ZERO;
        sample.pdf = (d * n_dot_h) / (4.f * h_dot_wo);
        sample.type = (instance->roughness < 0.02f) ? MaterialSample::Specular : MaterialSample::Glossy;

        if(sample.pdf < 0.01f)
            samples.pop_back();
    }

    return true;
}

bool MaterialGGXReflect::evaluate_material(const MaterialInstance * material_instance, MaterialSample &sample)
{
    const MaterialInstanceGGXReflect * instance = dynamic_cast<const MaterialInstanceGGXReflect *>(material_instance);

    if(sample.wo.dot(instance->surface->normal) < 0)
        return false;

    const Vec3f h = (sample.wo - instance->surface->wi).normalized();
    const float n_dot_h = clamp(h.dot(instance->surface->normal), -1.f, 1.f);
    const float h_dot_wi = clamp(h.dot(-instance->surface->wi), -1.f, 1.f);
    const float h_dot_wo = clamp(h.dot(sample.wo), -1.f, 1.f);
    const float n_dot_wi = instance->surface->n_dot_wi;
    const float n_dot_wo = clamp(instance->surface->normal.dot(sample.wo), -1.f, 1.f);

    const float d = D(instance->surface->normal, h, n_dot_h, instance->roughness_sqr);
    const float g = G1(-instance->surface->wi, instance->surface->normal, h, h_dot_wi, n_dot_wi, instance->roughness_sqr)*G1(sample.wo, instance->surface->normal, h, h_dot_wo, n_dot_wo, instance->roughness_sqr);
    const Vec3f f = instance->fresnel->Fr(fabs(h_dot_wi));

    sample.weight = (instance->specular_color * f * g * d) / (4.f * n_dot_wi);
    sample.pdf = (d * n_dot_h) / (4.f * h_dot_wo);

    return true;
}

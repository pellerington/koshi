#include "MaterialGGXReflect.h"

#include "../Math/RNG.h"
#include "../Math/Helpers.h"
#include "../Util/Color.h"
#include <cmath>
#include <iostream>

MaterialGGXReflect::MaterialGGXReflect(const AttributeVec3f &specular_color_attribute, const AttributeFloat &roughness_attribute, std::shared_ptr<Fresnel> fresnel)
: specular_color_attribute(specular_color_attribute), roughness_attribute(roughness_attribute), fresnel(fresnel)
{
}

std::shared_ptr<Material> MaterialGGXReflect::instance(const Surface * surface)
{
    std::shared_ptr<MaterialGGXReflect> material(new MaterialGGXReflect(*this));
    material->surface = surface;
    material->specular_color = specular_color_attribute.get_value(surface->u, surface->v, 0.f);
    material->roughness = roughness_attribute.get_value(surface->u, surface->v, 0.f);
    material->roughness = clamp(material->roughness * material->roughness, 0.01f, 0.99f);
    material->roughness_sqr = material->roughness * material->roughness;
    if(!fresnel) material->fresnel = std::shared_ptr<Fresnel>(new FresnelMetalic(material->specular_color));
    return material;
}

bool MaterialGGXReflect::sample_material(std::vector<MaterialSample> &samples, const float sample_reduction)
{
    if(!surface)
        return false;

    // Estimate the number of samples
    uint num_samples = SAMPLES_PER_SA * sqrtf(roughness);
    const float quality = 1.f / num_samples;
    num_samples = std::max(1.f, num_samples * sample_reduction);

    std::vector<Vec2f> rnd;
    RNG::Rand2d(num_samples, rnd);

    for(uint i = 0; i < rnd.size(); i++)
    {
        const float theta = TWO_PI * rnd[i][0];
        const float phi = atanf(roughness * sqrtf(rnd[i][1]) / sqrtf(1.f - rnd[i][1]));
        const Vec3f h = (surface->front ? 1.f : -1.f) * (surface->transform * Vec3f(sinf(phi) * cosf(theta), cosf(phi), sinf(phi) * sinf(theta)));
        const float h_dot_wi = clamp(h.dot(-surface->wi), -1.f, 1.f);

        // If we are inside the only time we want to call this is if we have total internal reflection.
        const Vec3f f = fresnel->Fr(fabs(h_dot_wi));
        if(!surface->front && f < 1.f)
            continue;

        samples.emplace_back();
        MaterialSample &sample = samples.back();
        sample.quality = quality;
        sample.wo = (2.f * h_dot_wi * h + surface->wi);

        const Vec3f normal = (surface->front ? 1.f : -1.f) * surface->normal;
        const float n_dot_h = clamp(h.dot(normal), -1.f, 1.f);
        const float h_dot_wo = clamp(h.dot(sample.wo), -1.f, 1.f);
        const float n_dot_wi = fabs(surface->n_dot_wi);
        const float n_dot_wo = clamp(normal.dot(sample.wo), -1.f, 1.f);

        const float d = D(normal, h, n_dot_h, roughness_sqr);
        const float g = G1(-surface->wi, normal, h, h_dot_wi, n_dot_wi, roughness_sqr)*G1(sample.wo, normal, h, h_dot_wo, n_dot_wo, roughness_sqr);

        sample.weight = (n_dot_wo > 0.f) ? (specular_color * f * g * d) / (4.f * n_dot_wi) : VEC3F_ZERO;
        sample.pdf = (d * n_dot_h) / (4.f * h_dot_wo);
        sample.type = (roughness < 0.02f) ? MaterialSample::Specular : MaterialSample::Glossy;

        if(sample.pdf < 0.01f)
            samples.pop_back();
    }

    return true;
}

bool MaterialGGXReflect::evaluate_material(MaterialSample &sample)
{
    if(!surface || sample.wo.dot(surface->normal) < 0)
        return false;

    const Vec3f h = (sample.wo - surface->wi).normalized();
    const float n_dot_h = clamp(h.dot(surface->normal), -1.f, 1.f);
    const float h_dot_wi = clamp(h.dot(-surface->wi), -1.f, 1.f);
    const float h_dot_wo = clamp(h.dot(sample.wo), -1.f, 1.f);
    const float n_dot_wi = surface->n_dot_wi;
    const float n_dot_wo = clamp(surface->normal.dot(sample.wo), -1.f, 1.f);

    const float d = D(surface->normal, h, n_dot_h, roughness_sqr);
    const float g = G1(-surface->wi, surface->normal, h, h_dot_wi, n_dot_wi, roughness_sqr)*G1(sample.wo, surface->normal, h, h_dot_wo, n_dot_wo, roughness_sqr);
    const Vec3f f = fresnel->Fr(fabs(h_dot_wi));

    sample.weight = (specular_color * f * g * d) / (4.f * n_dot_wi);
    sample.pdf = (d * n_dot_h) / (4.f * h_dot_wo);

    return true;
}

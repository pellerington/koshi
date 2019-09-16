#include "Dielectric.h"

#include "../Math/RNG.h"
#include "../Math/Helpers.h"
#include "../Util/Color.h"
#include <cmath>
#include <iostream>

Dielectric::Dielectric(Vec3f reflective_color, Vec3f refractive_color, float roughness, float ior, Vec3f emission)
: ior(ior), emission(emission)
{
    ggx = std::shared_ptr<GGXReflect>(new GGXReflect(reflective_color, roughness, fresnel));
    ggx_refract = std::shared_ptr<GGXRefract>(new GGXRefract(refractive_color, roughness, ior, fresnel));
}

std::shared_ptr<Material> Dielectric::instance(const Surface &surface)
{
    std::shared_ptr<Dielectric> material(new Dielectric(*this));
    // material.surface = surface;

    material->fresnel = (surface.enter) ? std::shared_ptr<Fresnel>(new FresnelDielectric(1.f, ior)) : std::shared_ptr<Fresnel>(new FresnelNone);
    material->ggx = std::dynamic_pointer_cast<GGXReflect>(material->ggx->instance(surface));
    material->ggx->set_fresnel(material->fresnel);
    material->ggx_refract = std::dynamic_pointer_cast<GGXRefract>(material->ggx_refract->instance(surface));
    material->ggx_refract->set_fresnel(material->fresnel);

    return material;
}

bool Dielectric::sample_material(const Surface &surface, std::deque<SrfSample> &srf_samples, float sample_reduction)
{
    if(surface.enter)
        ggx->sample_material(surface, srf_samples, sample_reduction);
    const float reflective_samples = srf_samples.size();
    ggx_refract->sample_material(surface, srf_samples, sample_reduction);
    const float refractive_samples = srf_samples.size() - reflective_samples;
    const float total_samples = reflective_samples + refractive_samples;

    const float reflective_weight = reflective_samples / total_samples;
    for(uint i = 0; i < reflective_samples; i++)
        srf_samples[i].pdf *= reflective_weight;
    const float refractive_weight = refractive_samples / total_samples;
    for(uint i = reflective_samples; i < total_samples; i++)
        srf_samples[i].pdf *= refractive_weight;

    return true;
}

bool Dielectric::evaluate_material(const Surface &surface, SrfSample &srf_sample, float &pdf)
{
    return false;
}

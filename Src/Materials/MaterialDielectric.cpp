#include "MaterialDielectric.h"

#include "../Math/RNG.h"
#include "../Math/Helpers.h"
#include "../Util/Color.h"
#include <cmath>
#include <iostream>

MaterialDielectric::MaterialDielectric(const Vec3f &reflective_color, const Vec3f &refractive_color, const float &roughness, const float &ior, const Vec3f &emission)
: ior(ior), emission(emission)
{
    ggx_reflect = std::shared_ptr<MaterialGGXReflect>(new MaterialGGXReflect(reflective_color, roughness, fresnel));
    ggx_refract = std::shared_ptr<MaterialGGXRefract>(new MaterialGGXRefract(refractive_color, roughness, ior, fresnel));
}

std::shared_ptr<Material> MaterialDielectric::instance(const Surface * surface)
{
    std::shared_ptr<MaterialDielectric> material(new MaterialDielectric(*this));
    material->surface = surface;

    const float ior_in = surface->ior.curr_ior;
    const float ior_out = surface->front ? ior : ((surface->ior.prev) ? surface->ior.prev->curr_ior : 1.f);

    material->fresnel = std::shared_ptr<Fresnel>(new FresnelDielectric(ior_in, ior_out));
    material->ggx_reflect = std::dynamic_pointer_cast<MaterialGGXReflect>(material->ggx_reflect->instance(surface));
    material->ggx_reflect->set_fresnel(material->fresnel);
    material->ggx_refract = std::dynamic_pointer_cast<MaterialGGXRefract>(material->ggx_refract->instance(surface));
    material->ggx_refract->set_fresnel(material->fresnel);

    return material;
}

bool MaterialDielectric::sample_material(std::vector<MaterialSample> &samples, const float sample_reduction)
{
    if(!surface)
        return false;

    ggx_reflect->sample_material(samples, sample_reduction);
    const float reflective_samples = samples.size();
    ggx_refract->sample_material(samples, sample_reduction);
    const float refractive_samples = samples.size() - reflective_samples;
    const float total_samples = reflective_samples + refractive_samples;

    const float reflective_weight = reflective_samples / total_samples;
    for(uint i = 0; i < reflective_samples; i++)
        samples[i].pdf *= reflective_weight;
    const float refractive_weight = refractive_samples / total_samples;
    for(uint i = reflective_samples; i < total_samples; i++)
        samples[i].pdf *= refractive_weight;

    return true;
}

bool MaterialDielectric::evaluate_material(MaterialSample &sample)
{
    if(!surface)
        return false;

    sample.weight = 0.f;
    sample.pdf = 0.f;

    MaterialSample isample = sample;
    if(surface->front)
        if(ggx_reflect->evaluate_material(isample))
        {
            sample.weight += isample.weight;
            sample.pdf += isample.pdf;
        }
    isample = sample;
    if(ggx_refract->evaluate_material(isample))
    {
        sample.weight += isample.weight;
        sample.pdf += isample.pdf;
    }

    return true;
}

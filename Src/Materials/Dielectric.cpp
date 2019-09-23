#include "Dielectric.h"

#include "../Math/RNG.h"
#include "../Math/Helpers.h"
#include "../Util/Color.h"
#include <cmath>
#include <iostream>

Dielectric::Dielectric(const Vec3f &reflective_color, const Vec3f &refractive_color, const float &roughness, const float &ior, const Vec3f &emission)
: ior(ior), emission(emission)
{
    ggx_reflect = std::shared_ptr<GGXReflect>(new GGXReflect(reflective_color, roughness, fresnel));
    ggx_refract = std::shared_ptr<GGXRefract>(new GGXRefract(refractive_color, roughness, ior, fresnel));
}

std::shared_ptr<Material> Dielectric::instance(const Surface &surface)
{
    std::shared_ptr<Dielectric> material(new Dielectric(*this));

    material->fresnel = (surface.enter) ? std::shared_ptr<Fresnel>(new FresnelDielectric(1.f, ior)) : std::shared_ptr<Fresnel>(new FresnelNone);
    material->ggx_reflect = std::dynamic_pointer_cast<GGXReflect>(material->ggx_reflect->instance(surface));
    material->ggx_reflect->set_fresnel(material->fresnel);
    material->ggx_refract = std::dynamic_pointer_cast<GGXRefract>(material->ggx_refract->instance(surface));
    material->ggx_refract->set_fresnel(material->fresnel);

    return material;
}

bool Dielectric::sample_material(const Surface &surface, std::deque<PathSample> &path_samples, const float sample_reduction)
{
    if(surface.enter)
        ggx_reflect->sample_material(surface, path_samples, sample_reduction);
    const float reflective_samples = path_samples.size();
    ggx_refract->sample_material(surface, path_samples, sample_reduction);
    const float refractive_samples = path_samples.size() - reflective_samples;
    const float total_samples = reflective_samples + refractive_samples;

    const float reflective_weight = reflective_samples / total_samples;
    for(uint i = 0; i < reflective_samples; i++)
        path_samples[i].pdf *= reflective_weight;
    const float refractive_weight = refractive_samples / total_samples;
    for(uint i = reflective_samples; i < total_samples; i++)
        path_samples[i].pdf *= refractive_weight;

    return true;
}

bool Dielectric::evaluate_material(const Surface &surface, PathSample &path_sample, float &pdf)
{
    path_sample.fr = 0.f;
    pdf = 0.f;

    float rpdf;
    PathSample rsample = path_sample;
    if(surface.enter)
        if(ggx_reflect->evaluate_material(surface, rsample, rpdf))
        {
            path_sample.fr += rsample.fr;
            pdf += rpdf;
        }
    rsample = path_sample;
    if(ggx_refract->evaluate_material(surface, rsample, rpdf))
    {
        path_sample.fr += rsample.fr;
        pdf += rpdf;
    }

    return true;
}

#include "MaterialDielectric.h"

#include "../Math/Helpers.h"
#include "../Util/Color.h"
#include <cmath>
#include <iostream>

MaterialDielectric::MaterialDielectric(const AttributeVec3f &reflective_color_attribute,
                                       const AttributeVec3f &refractive_color_attribute,
                                       const AttributeFloat &roughness_attribute,
                                       const float &ior)
: reflective_color_attribute(reflective_color_attribute), refractive_color_attribute(refractive_color_attribute),
  roughness_attribute(roughness_attribute), ior(ior),
  ggx_reflect(MaterialGGXReflect(reflective_color_attribute, roughness_attribute)),
  ggx_refract(MaterialGGXRefract(refractive_color_attribute, roughness_attribute, ior))
{
}

MaterialInstance * MaterialDielectric::instance(const Surface * surface, Resources &resources)
{
    MaterialInstanceDielectric * instance = resources.memory.create<MaterialInstanceDielectric>();

    instance->surface = surface;
    instance->reflect_instance.surface = surface;
    instance->refract_instance.surface = surface;

    instance->refract_instance.ior_in = surface->curr_ior;
    instance->refract_instance.ior_out = surface->front ? ior : surface->prev_ior;
    instance->refract_instance.refractive_color = refractive_color_attribute.get_value(surface->u, surface->v, 0.f);
    instance->refract_instance.roughness = roughness_attribute.get_value(surface->u, surface->v, 0.f);
    instance->refract_instance.roughness = clamp(instance->refract_instance.roughness * instance->refract_instance.roughness, 0.01f, 0.99f);
    instance->refract_instance.roughness_sqr = instance->refract_instance.roughness * instance->refract_instance.roughness;
    instance->refract_instance.fresnel = resources.memory.create<FresnelDielectric>(instance->refract_instance.ior_in, instance->refract_instance.ior_out);

    instance->reflect_instance.specular_color = reflective_color_attribute.get_value(surface->u, surface->v, 0.f);
    instance->reflect_instance.roughness = instance->refract_instance.roughness;
    instance->reflect_instance.roughness_sqr = instance->refract_instance.roughness_sqr;
    instance->reflect_instance.fresnel = instance->refract_instance.fresnel;

    return instance;
}

bool MaterialDielectric::sample_material(const MaterialInstance * material_instance, std::vector<MaterialSample> &samples, const float sample_reduction, Resources &resources)
{
    const MaterialInstanceDielectric * instance = (const MaterialInstanceDielectric *)material_instance;

    ggx_reflect.sample_material(&instance->reflect_instance, samples, sample_reduction, resources);
    const float reflective_samples = samples.size();
    ggx_refract.sample_material(&instance->refract_instance, samples, sample_reduction, resources);
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

bool MaterialDielectric::evaluate_material(const MaterialInstance * material_instance, MaterialSample &sample)
{
    const MaterialInstanceDielectric * instance = (const MaterialInstanceDielectric *)material_instance;

    sample.weight = 0.f;
    sample.pdf = 0.f;

    MaterialSample isample = sample;
    if(instance->surface->front)
        if(ggx_reflect.evaluate_material(&instance->reflect_instance, isample))
        {
            sample.weight += isample.weight;
            sample.pdf += isample.pdf;
        }
    isample = sample;
    if(ggx_refract.evaluate_material(&instance->refract_instance, isample))
    {
        sample.weight += isample.weight;
        sample.pdf += isample.pdf;
    }

    return true;
}

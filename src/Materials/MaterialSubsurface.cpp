#include <Materials/MaterialSubsurface.h>

#include <Math/Helpers.h>
#include <Util/Color.h>
#include <cmath>
#include <iostream>

MaterialSubsurface::MaterialSubsurface(const AttributeVec3f &surface_color_attr, const AttributeFloat &surface_weight_attr)
: surface_color_attr(surface_color_attr), surface_weight_attr(surface_weight_attr),
  lambert(MaterialLambert(surface_color_attr)), back_lambert(MaterialBackLambert(surface_color_attr))
{
}

MaterialInstance * MaterialSubsurface::instance(const Surface * surface, Resources &resources)
{
    MaterialInstanceSubsurface * instance = resources.memory.create<MaterialInstanceSubsurface>();
    instance->surface = surface;
    instance->surface_weight = surface_weight_attr.get_value(surface->u, surface->v, 0.f, resources);
    instance->lambert_instance.surface = surface;
    instance->lambert_instance.diffuse_color = surface_color_attr.get_value(surface->u, surface->v, 0.f, resources);
    instance->back_lambert_instance.surface = surface;
    instance->back_lambert_instance.diffuse_color = instance->lambert_instance.diffuse_color;
    return instance;
}

bool MaterialSubsurface::sample_material(const MaterialInstance * material_instance, std::vector<MaterialSample> &samples, const float sample_reduction, Resources &resources)
{
    const MaterialInstanceSubsurface * instance = (const MaterialInstanceSubsurface *)material_instance;

    if(instance->surface->front)
        lambert.sample_material(&instance->lambert_instance, samples, sample_reduction*instance->surface_weight, resources);
    const float front_samples = samples.size();
    back_lambert.sample_material(&instance->back_lambert_instance, samples, sample_reduction*(instance->surface->front ? 1.f-instance->surface_weight : 1.f), resources);
    const float back_samples = samples.size() - front_samples;
    const float total_samples = front_samples + back_samples;

    const float front_weight = front_samples / total_samples;
    for(uint i = 0; i < front_samples; i++)
        samples[i].pdf *= front_weight * (1.f / instance->surface_weight);
    const float back_weight = back_samples / total_samples;
    for(uint i = front_samples; i < total_samples; i++)
        samples[i].pdf *= back_weight * (1.f / (1.f - instance->surface_weight));

    return true;
}

bool MaterialSubsurface::evaluate_material(const MaterialInstance * material_instance, MaterialSample &sample)
{
    const MaterialInstanceSubsurface * instance = (const MaterialInstanceSubsurface *)material_instance;

    if(instance->surface->front)
        lambert.evaluate_material(&instance->lambert_instance, sample);
    else
        back_lambert.evaluate_material(&instance->back_lambert_instance, sample);

    return true;
}

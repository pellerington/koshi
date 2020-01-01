#include "MaterialSubsurface.h"

#include "../Math/Helpers.h"
#include "../Util/Color.h"
#include <cmath>
#include <iostream>

MaterialSubsurface::MaterialSubsurface(const AttributeVec3f &surface_color, const AttributeFloat &surface_weight_attr)
: surface_weight_attr(surface_weight_attr)
{
    lambert = std::shared_ptr<MaterialLambert>(new MaterialLambert(surface_color));
    back_lambert = std::shared_ptr<MaterialBackLambert>(new MaterialBackLambert(surface_color));
}

std::shared_ptr<Material> MaterialSubsurface::instance(const Surface * surface, RNG &rng)
{
    std::shared_ptr<MaterialSubsurface> material(new MaterialSubsurface(*this));
    material->surface = surface;
    material->rng = &rng;
    material->surface_weight = surface_weight_attr.get_value(surface->u, surface->v, 0.f);
    material->lambert = std::dynamic_pointer_cast<MaterialLambert>(material->lambert->instance(surface, rng));
    material->back_lambert = std::dynamic_pointer_cast<MaterialBackLambert>(material->back_lambert->instance(surface, rng));
    return material;
}

bool MaterialSubsurface::sample_material(std::vector<MaterialSample> &samples, const float sample_reduction)
{
    if(!surface)
        return false;

    if(surface->front)
        lambert->sample_material(samples, sample_reduction*surface_weight);
    const float front_samples = samples.size();
    back_lambert->sample_material(samples, sample_reduction*(surface->front ? 1.f-surface_weight : 1.f));
    const float back_samples = samples.size() - front_samples;
    const float total_samples = front_samples + back_samples;

    const float front_weight = front_samples / total_samples;
    for(uint i = 0; i < front_samples; i++)
        samples[i].pdf *= front_weight * (1.f / surface_weight);
    const float back_weight = back_samples / total_samples;
    for(uint i = front_samples; i < total_samples; i++)
        samples[i].pdf *= back_weight * (1.f / (1.f - surface_weight));

    return true;
}

bool MaterialSubsurface::evaluate_material(MaterialSample &sample)
{
    if(!surface)
        return false;

    if(surface->front)
        lambert->evaluate_material(sample);
    else
        back_lambert->evaluate_material(sample);

    return true;
}

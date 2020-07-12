#pragma once

#include <integrator/Integrator.h>
#include <material/Material.h>
#include <Util/Color.h>

#define SAMPLES_PER_SA 64

struct SurfaceSample
{
    Vec3f li;
    Vec3f weight;
    float pdf;
    const IntersectList * intersects;
    bool scatter = false;
};

class SurfaceSampler : public Integrator
{
public:
    void pre_render(Resources& resources)
    {
        min_quality = std::pow(1.f / SAMPLES_PER_SA, resources.settings->depth);
    }

    struct SurfaceSamplerData : public IntegratorData
    {
        const Surface * surface;
    };
    IntegratorData * pre_integrate(const Intersect * intersect, Resources& resources)
    {
        SurfaceSamplerData * data = resources.memory->create<SurfaceSamplerData>();
        data->surface = dynamic_cast<const Surface *>(intersect->geometry_data);
        return data;
    }

    Vec3f integrate(const Intersect * intersect, IntegratorData * data, Transmittance& transmittance, Resources& resources) const
    {
        // Check it is geometry surface.
        SurfaceSamplerData * surface_data = (SurfaceSamplerData *)data;
        const Surface * surface = surface_data->surface;
        if(!surface || !surface->material) return VEC3F_ZERO;

        Vec3f color = VEC3F_ZERO;

        // Add light contribution.
        color += (surface->facing) ? surface->material->emission(surface->u, surface->v, surface->w, intersect, resources) : VEC3F_ZERO;

        // Terminate before scattering?
        if(is_saturated(color) || intersect->path->depth > resources.settings->max_depth || intersect->path->quality < min_quality)
            return color * transmittance.shadow(intersect->t, resources) * surface->opacity;

        MaterialLobes lobes = surface->material->instance(surface, intersect, resources);
        if(!lobes.size()) 
            return color * transmittance.shadow(intersect->t, resources) * surface->opacity; 

        Interiors interiors(intersect->t, transmittance.get_intersects());

        Array<SurfaceSample> samples(resources.memory);
        scatter_surface(samples, lobes, intersect, surface_data, interiors, resources);
        for(uint i = 0; i < samples.size(); i++)
        {
            color += samples[i].li * samples[i].weight / samples[i].pdf;
        }

        return color * transmittance.shadow(intersect->t, resources) * surface->opacity;
    }

    virtual Vec3f shadow(const float& t, const Intersect * intersect, IntegratorData * data, Resources& resources) const
    {
        SurfaceSamplerData * surface_data = (SurfaceSamplerData *)data;
        return (t > intersect->t) ? (VEC3F_ONES - surface_data->surface->opacity) : VEC3F_ONES;
    }

    virtual void scatter_surface(
        Array<SurfaceSample>& samples,
        const MaterialLobes& lobes,
        const Intersect * intersect, SurfaceSamplerData * data, 
        Interiors& interiors, Resources& resources) const = 0;

    virtual float evaluate(const SurfaceSample& sample, 
        const MaterialLobes& lobes,
        const Intersect * intersect, SurfaceSamplerData * data, 
        Resources& resources) const = 0;

private:
    float min_quality;
};
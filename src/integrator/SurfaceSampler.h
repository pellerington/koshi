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
        Vec3f color = VEC3F_ZERO;

        // Check it is geometry surface.
        SurfaceSamplerData * surface_data = (SurfaceSamplerData *)data;
        const Surface * surface = surface_data->surface;
        if(!surface) return color;

        // Add light contribution.
        Light * light = intersect->geometry->get_attribute<Light>("light");
        if(light && surface->facing) color += light->get_intensity(surface->u, surface->v, 0.f, intersect, resources);

        // TODO: Place the material on the geometry surface so it can be overidden.
        Material * material = intersect->geometry->get_attribute<Material>("material");

        if(is_saturated(color) || intersect->path->depth > resources.settings->max_depth || intersect->path->quality < min_quality || !material)
        {
            return color * transmittance.shadow(intersect->t, resources) * surface->opacity;
        }

        MaterialInstance material_instance = material->instance(surface, intersect, resources);

        Interiors interiors(intersect->t, transmittance.get_intersects());

        Array<SurfaceSample> samples(resources.memory);
        scatter_surface(samples, material_instance, intersect, surface_data, interiors, resources);
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
        const MaterialInstance& material_instance,
        const Intersect * intersect, SurfaceSamplerData * data, 
        Interiors& interiors, Resources& resources) const = 0;

    virtual float evaluate(const SurfaceSample& sample, 
        const MaterialInstance& material_instance,
        const Intersect * intersect, SurfaceSamplerData * data, 
        Resources& resources) const = 0;

private:
    float min_quality;
};
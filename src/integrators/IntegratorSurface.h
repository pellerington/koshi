#pragma once

#include <integrators/Integrator.h>
#include <materials/Material.h>
#include <Util/Color.h>

#define SAMPLES_PER_SA 64

struct SurfaceSample
{
    Vec3f li;
    Vec3f weight;
    float pdf;
    const IntersectList * intersects;
};

class IntegratorSurface : public Integrator
{
public:
    Vec3f integrate(const Intersect * intersect, Transmittance& transmittance, Resources &resources) const
    {
        Vec3f color = VEC3F_ZERO;

        // Check it is geometry surface.
        // dynamic_cast<GeometrySurface*>
        const GeometrySurface * surface = &intersect->surface;

        // Add light contribution.
        Light * light = intersect->geometry->get_attribute<Light>("light");
        if(light) color += light->get_intensity(intersect, resources);

        // TODO: put this in the pre_render step (no need to recalculate every time)
        const float min_quality = std::pow(1.f / SAMPLES_PER_SA, resources.settings->depth);

        if(is_saturated(color) || intersect->path->depth > resources.settings->max_depth || intersect->path->quality < min_quality)
            return color * transmittance.shadow(intersect->t);

        Material * material = intersect->geometry->get_attribute<Material>("material");
        if(!material) return color * transmittance.shadow(intersect->t);

        MaterialInstance material_instance = material->instance(surface, resources);

        std::vector<SurfaceSample> scatter = integrate_surface(material_instance, intersect, surface, resources);
        for(uint i = 0; i < scatter.size(); i++)
        {
            color += scatter[i].li * scatter[i].weight / scatter[i].pdf;
        }

        return color * transmittance.shadow(intersect->t);
    }

    virtual std::vector<SurfaceSample> integrate_surface(
        const MaterialInstance& material_instance,
        const Intersect * intersect, const GeometrySurface * surface, 
        Resources& resources) const = 0;

    virtual float evaluate(const SurfaceSample& sample, 
        const MaterialInstance& material_instance,
        const Intersect * intersect, const GeometrySurface * surface, 
        Resources& resources) const = 0;
};
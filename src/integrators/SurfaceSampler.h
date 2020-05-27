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

class SurfaceSampler : public Integrator
{
public:
    Vec3f integrate(const Intersect * intersect, Transmittance& transmittance, Resources &resources) const
    {
        Vec3f color = VEC3F_ZERO;

        // Check it is geometry surface.
        const GeometrySurface * surface = dynamic_cast<const GeometrySurface*>(intersect->geometry_data);
        if(!surface) return color;

        // Add light contribution.
        Light * light = intersect->geometry->get_attribute<Light>("light");
        if(light && surface->facing) color += light->get_intensity(surface->u, surface->v, 0.f, intersect, resources);

        // TODO: put this in the pre_render step (no need to recalculate every time)
        const float min_quality = std::pow(1.f / SAMPLES_PER_SA, resources.settings->depth);

        // TODO: Place the material on the geometry surface so it can be overidden.
        Material * material = intersect->geometry->get_attribute<Material>("material");

        if(is_saturated(color) || intersect->path->depth > resources.settings->max_depth || intersect->path->quality < min_quality || !material)
        {
            return color * transmittance.shadow(intersect->t) * surface->opacity;
        }

        MaterialInstance material_instance = material->instance(surface, resources);

        Interiors interiors(intersect->t, transmittance.get_intersects());

        std::vector<SurfaceSample> scatter = integrate_surface(material_instance, intersect, surface, interiors, resources);
        for(uint i = 0; i < scatter.size(); i++)
        {
            color += scatter[i].li * scatter[i].weight / scatter[i].pdf;
        }

        return color * transmittance.shadow(intersect->t) * surface->opacity;
    }

    virtual Vec3f shadow(const float& t, const Intersect * intersect) const
    {
        // TODO: Calculate shadow and get surface in the pre_integrate step!
        // TODO: Have an optional "shadow" and opacity.
        const GeometrySurface * surface = dynamic_cast<const GeometrySurface*>(intersect->geometry_data);
        return (t > intersect->t) ? ((surface) ? (VEC3F_ONES - surface->opacity) : VEC3F_ZERO) : VEC3F_ONES;
    }

    virtual std::vector<SurfaceSample> integrate_surface(
        const MaterialInstance& material_instance,
        const Intersect * intersect, const GeometrySurface * surface, 
        Interiors& interiors, Resources& resources) const = 0;

    virtual float evaluate(const SurfaceSample& sample, 
        const MaterialInstance& material_instance,
        const Intersect * intersect, const GeometrySurface * surface, 
        Resources& resources) const = 0;
};
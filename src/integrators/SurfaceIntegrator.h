#pragma once

#include <integrators/Integrator.h>
#include <Materials/Material.h>
#include <Util/Color.h>

struct SurfaceSample
{
    // TODO: Passing the intersects this way is anoying and horrible. Find a better way.
    SurfaceSample(const IntersectList& intersects) : intersects(intersects) {}

    Vec3f li;
    float weight;
    float pdf;
    MaterialSample material_sample;
    IntersectList intersects;
};

class SurfaceIntegrator : public Integrator
{
public:
    Vec3f integrate(const Intersect& intersect/*, Transmittance& transmittance*/, Resources &resources) const
    {
        Vec3f color = VEC3F_ZERO;

        // Check it is geometry surface.
        // dynamic_cast<GeometrySurface*>
        const GeometrySurface * surface = &intersect.surface;

        // Add light contribution.
        Light * light = intersect.geometry->get_attribute<Light>("light");
        if(light) color += light->get_intensity(intersect, resources);

        // TODO: put this in the preintersect.geometry->get_attribute<Material>("material");_render step (no need to recalculate every time)
        float min_quality = std::pow(1.f / SAMPLES_PER_SA, resources.settings->depth);

        if(is_saturated(color) || intersect.path->depth > resources.settings->max_depth || intersect.path->quality < min_quality)
            return color;

        Material * material = intersect.geometry->get_attribute<Material>("material");
        if(!material) return color;

        MaterialInstance * material_instance = material->instance(surface, resources);

        std::vector<SurfaceSample> scatter = integrate_surface(material_instance, material, intersect, surface, resources);
        for(uint i = 0; i < scatter.size(); i++)
            color += scatter[i].li * scatter[i].material_sample.weight * scatter[i].weight / (scatter[i].pdf);

        return color;
    }

    virtual std::vector<SurfaceSample> integrate_surface(
        MaterialInstance * material_instance, Material * material, 
        const Intersect& intersect, const GeometrySurface * surface,
        Resources& resources) const = 0;

    virtual float evaluate(const Intersect& intersect, const SurfaceSample& sample, Resources& resources) = 0;
};
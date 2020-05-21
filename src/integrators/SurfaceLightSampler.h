#pragma once

#include <integrators/IntegratorSurface.h>
#include <lights/LightSampler.h>
#include <map>

class SurfaceLightSampler : public IntegratorSurface
{
public:
    void pre_render(Scene * scene) override;

    std::vector<SurfaceSample> integrate_surface(
        const MaterialInstance& material_instance,
        const Intersect * intersect, const GeometrySurface * surface, 
        InteriorMedium& interiors, Resources& resources) const;

    float evaluate(const SurfaceSample& sample, 
        const MaterialInstance& material_instance,
        const Intersect * intersect, const GeometrySurface * surface, 
        Resources& resources) const;

private:
    std::unordered_map<Geometry*, LightSampler*> lights;
};

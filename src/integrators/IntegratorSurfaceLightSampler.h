#pragma once

#include <integrators/IntegratorSurface.h>
#include <lights/LightSampler.h>
#include <map>

class IntegratorSurfaceLightSampler : public IntegratorSurface
{
public:
    void pre_render(Scene * scene) override;

    std::vector<SurfaceSample> integrate_surface(
        MaterialInstance * material_instance, Material * material, 
        const Intersect& intersect, const GeometrySurface * surface, 
        Resources& resources) const;

    float evaluate(const Intersect& intersect, const SurfaceSample& sample, Resources& resources);

private:
    std::unordered_map<Geometry*, LightSampler*> lights;
};

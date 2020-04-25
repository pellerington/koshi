#include <integrators/IntegratorSurfaceMaterialSampler.h>

std::vector<SurfaceSample> IntegratorSurfaceMaterialSampler::integrate_surface(
    MaterialInstance * material_instance, Material * material, 
    const Intersect& intersect, const GeometrySurface * surface, 
    Resources& resources) const
{
    const PathData * prev_path = intersect.path;
    const uint depth = prev_path ? prev_path->depth : 0;
    const float quality = prev_path ? prev_path->quality : 1.f;

    // TODO Make this adaptive sampling / have each bxdf call sample over and over.
    std::vector<MaterialSample> material_samples;
    material->sample_material(material_instance, material_samples, quality * resources.settings->sampling_quality, resources);

    std::vector<SurfaceSample> samples;
    const uint num_samples = material_samples.size();
    const float inv_num_samples = 1.f / (float)num_samples;
    for(uint i = 0; i < num_samples; i++)
    {
        PathData next_path;
        next_path.depth = depth + 1;
        next_path.quality = quality / num_samples;
        next_path.prev_path = prev_path;

        const bool reflect = material_samples[i].wo.dot(intersect.surface.normal) > 0;
        Ray ray(reflect ? intersect.surface.front_position : intersect.surface.back_position, material_samples[i].wo);
        samples.emplace_back(resources.intersector->intersect(ray, &next_path));
        
        // samples.back().intersects = resources.intersector->intersect(ray, &next_path);

        samples.back().material_sample = material_samples[i];
        samples.back().weight = inv_num_samples;
        samples.back().pdf = material_samples[i].pdf;

        samples.back().li = Integrator::shade(samples[i].intersects, resources);
    }

    return samples;
}

float IntegratorSurfaceMaterialSampler::evaluate(const Intersect& intersect, const SurfaceSample& sample, Resources& resources)
{
    return sample.material_sample.pdf;
}

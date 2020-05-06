#include <integrators/IntegratorSurfaceMaterialSampler.h>

std::vector<SurfaceSample> IntegratorSurfaceMaterialSampler::integrate_surface(
    const MaterialInstance& material_instance,
    const Intersect& intersect, const GeometrySurface * surface, 
    Resources& resources) const
{
    const PathData * prev_path = intersect.path;
    const uint depth = prev_path ? prev_path->depth : 0;
    const float quality = prev_path ? prev_path->quality : 1.f;

    std::vector<SurfaceSample> samples;
    for(uint l = 0; l < material_instance.size(); l++)
    {
        const MaterialLobe * lobe = material_instance[l];

        float next_quality = quality;
        uint num_samples = 1;
        if(quality < 1.f)
        {
            MaterialLobe::Type lobe_type = lobe->type();
            if(lobe_type == MaterialLobe::Diffuse)
            {
                num_samples = std::max(1u, (uint)(SAMPLES_PER_SA * quality * resources.settings->sampling_quality));
                next_quality *= 1.f / SAMPLES_PER_SA;
            }
            else if(lobe_type == MaterialLobe::Glossy)
            {
                const float num_glossy_samples = SAMPLES_PER_SA * sqrtf(lobe->roughness);
                num_samples = std::max(1u, (uint)(num_glossy_samples * quality * resources.settings->sampling_quality));
                next_quality *= 1.f / num_glossy_samples;
            }
        }

        const float inv_num_samples = 1.f / (float)num_samples;
        
        for(uint i = 0; i < num_samples; i++)
        {
            MaterialSample material_sample;
            if(!lobe->sample(material_sample, resources))
                continue;

            PathData next_path;
            next_path.depth = depth + 1;
            next_path.quality = next_quality;
            next_path.prev_path = prev_path;

            const bool reflect = material_sample.wo.dot(intersect.surface.normal) > 0;
            Ray ray(reflect ? intersect.surface.front_position : intersect.surface.back_position, material_sample.wo);
            samples.emplace_back(resources.intersector->intersect(ray, &next_path));
            
            SurfaceSample& sample = samples.back();

            // TODO: We need to be able to copy intersects
            // sample.intersects = resources.intersector->intersect(ray, &next_path);

            sample.material_sample = material_sample;

            sample.weight = inv_num_samples;
            sample.pdf = material_sample.pdf;

            // TODO: Do we need to evaluate pdf and weight for all other lobes? Maybe not

            sample.li = Integrator::shade(samples[i].intersects, resources);
        }
    }

    return samples;
}

float IntegratorSurfaceMaterialSampler::evaluate(const SurfaceSample& sample, 
    const MaterialInstance& material_instance,
    const Intersect& intersect, const GeometrySurface * surface, 
    Resources& resources) const
{
    return sample.material_sample.pdf;
}

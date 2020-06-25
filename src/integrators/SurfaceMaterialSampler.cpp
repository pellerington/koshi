#include <integrators/SurfaceMaterialSampler.h>
#include <intersection/Intersector.h>

std::vector<SurfaceSample> SurfaceMaterialSampler::integrate_surface(
    const MaterialInstance& material_instance,
    const Intersect * intersect, const Surface * surface, 
    Interiors& interiors, Resources& resources) const
{
    const PathData * prev_path = intersect->path;
    const uint depth = prev_path ? prev_path->depth : 0;
    const float quality = prev_path ? prev_path->quality : 1.f;

    // TODO: vector uses malloc which we should be avoiding. create an allocator using resources, OR make our won vector class.
    std::vector<SurfaceSample> samples;
    for(uint l = 0; l < material_instance.size(); l++)
    {
        const MaterialLobe * lobe = material_instance[l];

        float next_quality = quality;
        float num_unreduced_samples = 1.f;
        MaterialLobe::ScatterType scatter_type = lobe->get_scatter_type();
        if(scatter_type == MaterialLobe::DIFFUSE)
            num_unreduced_samples = SAMPLES_PER_SA;
        else if(scatter_type == MaterialLobe::GLOSSY)
            num_unreduced_samples = SAMPLES_PER_SA * sqrtf(lobe->roughness);
        next_quality *= 1.f / num_unreduced_samples;
        uint num_samples = std::max(1.f, num_unreduced_samples * quality * resources.settings->sampling_quality);

        MaterialLobe::Hemisphere hemisphere = lobe->get_hemisphere();

        // TODO: Make interiors better! Use hemisphere knowledge to decide if we want to do this.
        // transmit interiors
        Interiors transmit_interiors = surface->facing 
            ? interiors.push(intersect->geometry, lobe->interior)
            : interiors.pop(intersect->geometry);

        const float inv_num_samples = 1.f / (float)num_samples;
        for(uint i = 0; i < num_samples; i++)
        {
            MaterialSample material_sample;
            if(!lobe->sample(material_sample, resources))
                continue;

            const bool front = material_sample.wo.dot(surface->normal) > 0;
            const bool transmit = front ^ surface->facing;

            if((transmit && hemisphere == MaterialLobe::FRONT) || (!transmit && hemisphere == MaterialLobe::BACK))
                continue;

            PathData next_path;
            next_path.depth = depth + 1;
            next_path.quality = next_quality;
            next_path.prev_path = prev_path;

            const Vec3f& ray_position = front ? surface->front_position : surface->back_position;
            Ray ray(ray_position, material_sample.wo);

            samples.emplace_back();
            SurfaceSample& sample = samples.back();

            // TODO: Make this pre_intersect callback nicer
            sample.intersects = resources.intersector->intersect(ray, &next_path, resources, 
                Interiors::pre_intersect_callback, transmit ? &transmit_interiors : &interiors);
            sample.weight = material_sample.weight * inv_num_samples;
            sample.pdf = material_sample.pdf;

            // TODO: Do we need to evaluate pdf and weight for all other lobes? Probably?

            sample.li = Integrator::shade(sample.intersects, resources);
        }
    }

    return samples;
}

float SurfaceMaterialSampler::evaluate(const SurfaceSample& sample, 
    const MaterialInstance& material_instance,
    const Intersect * intersect, const Surface * surface, 
    Resources& resources) const
{
    float pdf = 0.f;
    const Vec3f& wo = intersect->ray.dir;
    for(uint i = 0; i < material_instance.size(); i++)
        pdf += material_instance[i]->pdf(wo, resources);
    return pdf;
}

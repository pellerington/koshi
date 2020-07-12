#include <integrator/SurfaceMaterialSampler.h>
#include <intersection/Intersector.h>
#include <math/Helpers.h>

void SurfaceMaterialSampler::scatter_surface(
    Array<SurfaceSample>& samples,
    const MaterialLobes& lobes,
    const Intersect * intersect, SurfaceSamplerData * data,
    Interiors& interiors, Resources& resources) const
{
    const Surface * surface = data->surface;
    const uint depth = intersect->path ? intersect->path->depth : 0;
    const float quality = intersect->path ? intersect->path->quality : 1.f;

    for(uint l = 0; l < lobes.size(); l++)
    {
        const MaterialLobe * lobe = lobes[l];

        MaterialLobe::ScatterType scatter_type = lobe->get_scatter_type();
        float num_samples = 0.f;
        const float min_num_samples = 4.f;
        float max_num_samples = 0.f;
        if(scatter_type == MaterialLobe::DIFFUSE)
            max_num_samples = SAMPLES_PER_SA;
        else if(scatter_type == MaterialLobe::GLOSSY)
            max_num_samples = SAMPLES_PER_SA * sqrtf(lobe->roughness);
        const float sample_quality = quality / max_num_samples;
        max_num_samples = std::max(1.f, max_num_samples * quality * resources.settings->sampling_quality);
        Vec3f variance_sum = VEC3F_ZERO, variance_sum_sqr = VEC3F_ZERO;
        const uint samples_begin = samples.size();

        MaterialLobe::Hemisphere hemisphere = lobe->get_hemisphere();

        // TODO: Make interiors better! Use hemisphere knowledge to decide if we want to do this.
        // transmit interiors
        Interiors transmit_interiors = surface->facing 
            ? interiors.push(intersect->geometry, lobe->interior)
            : interiors.pop(intersect->geometry);

        PathData next_path;
        next_path.depth = depth + 1;
        next_path.quality = sample_quality;
        next_path.prev_path = intersect->path;

        while(num_samples < max_num_samples && (num_samples < min_num_samples || variance(variance_sum, variance_sum_sqr, num_samples) > 0.05f*0.05f))
        {
            num_samples++;

            MaterialSample material_sample;
            if(!lobe->sample(material_sample, resources))
                continue;

            const bool front = material_sample.wo.dot(surface->normal) > 0;
            const bool transmit = front ^ surface->facing;

            if((transmit && hemisphere == MaterialLobe::FRONT) || (!transmit && hemisphere == MaterialLobe::BACK))
                continue;

            const Vec3f& ray_position = front ? surface->front_position : surface->back_position;
            Ray ray(ray_position, material_sample.wo);

            SurfaceSample& sample = samples.push();

            sample.intersects = resources.intersector->intersect(ray, &next_path, resources, 
                transmit ? transmit_interiors.get_callback() : interiors.get_callback());
            sample.weight = material_sample.weight;
            sample.pdf = material_sample.pdf;

            // TODO: Do we need to evaluate pdf and weight for all other lobes?

            sample.li = Integrator::shade(sample.intersects, resources);

            Vec3f color = (sample.li * sample.weight) / sample.pdf;
            variance_sum += color;
            variance_sum_sqr += color*color;
        }

        float inv_num_samples = 1.f / num_samples;
        for(uint i = samples_begin; i < samples.size(); i++)
            samples[i].weight *= inv_num_samples;
    }
}

float SurfaceMaterialSampler::evaluate(
    const SurfaceSample& sample, 
    const MaterialLobes& lobes,
    const Intersect * intersect, SurfaceSamplerData * data,
    Resources& resources) const
{
    if(sample.scatter)
        return false;
    float pdf = 0.f;
    const Vec3f& wo = intersect->ray.dir;
    for(uint i = 0; i < lobes.size(); i++)
        pdf += lobes[i]->pdf(wo, resources);
    return pdf;
}

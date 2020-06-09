#include <integrators/SurfaceLightSampler.h>
#include <base/Scene.h>
#include <intersection/Intersector.h>

void SurfaceLightSampler::pre_render(Resources& resources)
{
    for(auto object = resources.scene->begin(); object != resources.scene->end(); ++object)
    {
        Geometry * geometry = dynamic_cast<Geometry*>(object->second);
        if(geometry)
        {
            LightSampler * sampler = geometry->get_attribute<LightSampler>("light_sampler");
            if(sampler) lights[geometry] = sampler;
        }
    }
}

std::vector<SurfaceSample> SurfaceLightSampler::integrate_surface(
    const MaterialInstance& material_instance,
    const Intersect * intersect, const Surface * surface, 
    Interiors& interiors, Resources& resources) const
{
    const PathData * prev_path = intersect->path;
    const uint depth = prev_path ? prev_path->depth : 0;
    const float quality = prev_path ? prev_path->quality : 1.f;

    Interiors transmit_interiors = interiors.pop(intersect->geometry);

    std::vector<SurfaceSample> samples;
    for(auto light = lights.begin(); light != lights.end(); ++light)
    {
        // Make a better num_samples estimator
        uint num_samples = std::max(1.f, SAMPLES_PER_SA * quality * resources.settings->sampling_quality);
        std::vector<LightSample> light_samples;
        light->second->sample_light(num_samples, surface, light_samples, resources);
        num_samples = light_samples.size();
        const float inv_num_samples = 1.f / (float)num_samples;
        for(uint i = 0; i < num_samples; i++)
        {
            PathData next_path;
            next_path.depth = depth + 1;
            next_path.quality = quality / num_samples;
            next_path.prev_path = prev_path;

            Vec3f wo = (light_samples[i].position - surface->position).normalized();
            Vec3f weight;
            for(size_t i = 0; i < material_instance.size(); i++)
                weight += material_instance[i]->weight(wo, resources);

            if(is_black(weight)) continue;

            const bool front = wo.dot(surface->normal) > 0;
            const bool transmit = front ^ surface->facing;
            const Vec3f& ray_position = front ? surface->front_position : surface->back_position;
            Ray ray(ray_position, wo, 0.f, (light_samples[i].position - ray_position).length() - 2.f*RAY_OFFSET);

            samples.emplace_back();
            SurfaceSample& sample = samples.back();

            sample.intersects = resources.intersector->intersect(ray, &next_path, resources,
                Interiors::pre_intersect_callback, transmit ? &transmit_interiors : &interiors);

            Transmittance transmittance = Integrator::shadow(sample.intersects, resources);
            Vec3f shadow = transmittance.shadow(ray.tmax);

            sample.li = shadow * light_samples[i].intensity;
            sample.weight = weight * inv_num_samples;
            sample.pdf = light_samples[i].pdf;
        }
    }

    return samples;
}

float SurfaceLightSampler::evaluate(const SurfaceSample& sample, 
    const MaterialInstance& material_instance,
    const Intersect * intersect, const Surface * surface, 
    Resources& resources) const
{
    float pdf = 0.f;
    for(uint i = 0; i < sample.intersects->size(); i++)
    {
        const Intersect * light_intersect = sample.intersects->get(i);
        auto light = lights.find(light_intersect->geometry);
        if(light == lights.end()) continue;
        pdf += light->second->evaluate_light(light_intersect, surface, resources);
    }
    return pdf;
}

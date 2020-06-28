#include <integrator/SurfaceLightSampler.h>
#include <base/Scene.h>
#include <intersection/Intersector.h>

void SurfaceLightSampler::pre_render(Resources& resources)
{
    SurfaceSampler::pre_render(resources);

    for(auto object = resources.scene->begin(); object != resources.scene->end(); ++object)
    {
        Geometry * geometry = dynamic_cast<Geometry*>(object->second);
        if(geometry)
        {
            LightSampler * sampler = geometry->get_attribute<LightSampler>("light_sampler");
            if(sampler)
            {
                lights.push_back(sampler);
                lights_map[geometry] = lights.size() - 1;
            }
        }
    }
}

IntegratorData * SurfaceLightSampler::pre_integrate(const Intersect * intersect, Resources& resources)
{
    SurfaceLightSamplerData * data = resources.memory->create<SurfaceLightSamplerData>(resources);
    data->surface = dynamic_cast<const Surface *>(intersect->geometry_data);
    for(uint i = 0; i < lights.size(); i++)
        data->light_data.push(nullptr);
    return data;
}

void SurfaceLightSampler::scatter_surface(
    Array<SurfaceSample>& samples,
    const MaterialInstance& material_instance,
    const Intersect * intersect, SurfaceSamplerData * data, 
    Interiors& interiors, Resources& resources) const
{
    SurfaceLightSamplerData * sampler_data = (SurfaceLightSamplerData *)data;
    const Surface * surface = data->surface;
    const PathData * prev_path = intersect->path;
    const uint depth = prev_path ? prev_path->depth : 0;
    const float quality = prev_path ? prev_path->quality : 1.f;

    Interiors transmit_interiors = interiors.pop(intersect->geometry);

    for(uint l = 0; l < lights.size(); l++)
    {
        const LightSampler * light_sampler = lights[l];
        const LightSamplerData * light_sampler_data = sampler_data->light_data[l];
        if(!light_sampler_data)
            light_sampler_data = sampler_data->light_data[l] = light_sampler->pre_integrate(surface, resources);

        // TODO: Replace num_samples estimator with adaptive sampling.
        LightSampler::LightType light_type = light_sampler->get_light_type();
        uint num_samples = (light_type == LightSampler::POINT) ? 1u : std::max(1.f, SAMPLES_PER_SA * quality * resources.settings->sampling_quality);
        const float inv_num_samples = 1.f / (float)num_samples;

        for(uint i = 0; i < num_samples; i++)
        {
            PathData next_path;
            next_path.depth = depth + 1;
            next_path.quality = quality / num_samples;
            next_path.prev_path = prev_path;

            LightSample light_sample;
            if(!light_sampler->sample(light_sample, light_sampler_data, resources))
                continue;

            Vec3f wo = (light_sample.position - surface->position).normalized();
            const bool front = wo.dot(surface->normal) > 0;
            const bool transmit = front ^ surface->facing;

            Vec3f weight;
            for(size_t i = 0; i < material_instance.size(); i++)
            {
                MaterialLobe::Hemisphere hemisphere = material_instance[i]->get_hemisphere();
                if(hemisphere == MaterialLobe::SPHERE || (!transmit && hemisphere == MaterialLobe::FRONT) || (transmit && hemisphere == MaterialLobe::BACK))
                    weight += material_instance[i]->weight(wo, resources);
            }

            if(is_black(weight)) continue;

            const Vec3f& ray_position = front ? surface->front_position : surface->back_position;
            Ray ray(ray_position, wo, 0.f, (light_sample.position - ray_position).length() - 2.f*RAY_OFFSET);

            SurfaceSample& sample = samples.push();

            sample.intersects = resources.intersector->intersect(ray, &next_path, resources,
                Interiors::pre_intersect_callback, transmit ? &transmit_interiors : &interiors);

            Transmittance transmittance = Integrator::shadow(sample.intersects, resources);
            Vec3f shadow = transmittance.shadow(ray.tmax, resources);

            sample.li = shadow * light_sample.intensity;
            sample.weight = weight * inv_num_samples;
            sample.pdf = light_sample.pdf;
        }
    }
}

float SurfaceLightSampler::evaluate(
    const SurfaceSample& sample, 
    const MaterialInstance& material_instance,
    const Intersect * intersect, SurfaceSamplerData * data, 
    Resources& resources) const
{
    SurfaceLightSamplerData * sampler_data = (SurfaceLightSamplerData *)data;
    float pdf = 0.f;
    for(uint i = 0; i < sample.intersects->size(); i++)
    {
        const Intersect * light_intersect = sample.intersects->get(i);
        auto light_index = lights_map.find(light_intersect->geometry);
        if(light_index == lights_map.end()) continue;

        const LightSampler * light_sampler = lights[light_index->second];
        const LightSamplerData * light_sampler_data = sampler_data->light_data[light_index->second];
        if(!light_sampler_data)
            light_sampler_data = sampler_data->light_data[light_index->second] = light_sampler->pre_integrate(sampler_data->surface, resources);

        pdf += light_sampler->evaluate(light_intersect, light_sampler_data, resources);
    }
    return pdf;
}

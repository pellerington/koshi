#include <koshi/integrator/SurfaceLightSampler.h>
#include <koshi/base/Scene.h>
#include <koshi/intersection/Intersector.h>
#include <koshi/math/Helpers.h>

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
    const MaterialLobes& lobes,
    const Intersect * intersect, SurfaceSamplerData * data, 
    Interiors& interiors, Resources& resources) const
{
    SurfaceLightSamplerData * sampler_data = (SurfaceLightSamplerData *)data;
    const Surface * surface = data->surface;
    const uint depth = intersect->path ? intersect->path->depth : 0;
    const float quality = intersect->path ? intersect->path->quality : 1.f;

    Interiors transmit_interiors = interiors.pop(intersect->geometry);

    for(uint l = 0; l < lights.size(); l++)
    {
        const LightSampler * light_sampler = lights[l];
        const LightSamplerData * light_sampler_data = sampler_data->light_data[l];
        if(!light_sampler_data)
            light_sampler_data = sampler_data->light_data[l] = light_sampler->pre_integrate(surface, resources);

        LightSampler::LightType light_type = light_sampler->get_light_type();
        float num_samples = 0.f;
        const float min_num_samples = 4.f;
        const float max_num_samples = (light_type == LightSampler::POINT) ? 1.f : std::max(1.f, SAMPLES_PER_SA * quality * resources.settings->sampling_quality);
        Vec3f variance_sum = VEC3F_ZERO;
        const uint samples_begin = samples.size();

        PathData next_path;
        next_path.depth = depth + 1;
        next_path.quality = quality / num_samples;
        next_path.prev_path = intersect->path;

        while(num_samples < max_num_samples)
        {
            num_samples++;

            LightSample light_sample;
            if(!light_sampler->sample(light_sample, light_sampler_data, resources))
                continue;

            Vec3f wo = (light_sample.position - surface->position).normalized();
            const bool front = wo.dot(surface->normal) > 0.f;
            const bool transmit = front ^ surface->facing;

            Vec3f weight;
            for(size_t i = 0; i < lobes.size(); i++)
            {
                MaterialLobe::Hemisphere hemisphere = lobes[i]->get_hemisphere();
                if(hemisphere == MaterialLobe::SPHERE || (!transmit && hemisphere == MaterialLobe::FRONT) || (transmit && hemisphere == MaterialLobe::BACK))
                    weight += lobes[i]->weight(wo, resources);
            }

            if(is_black(weight)) continue;

            const Vec3f& ray_position = front ? surface->front_position : surface->back_position;
            Ray ray(ray_position, wo, 0.f, (light_sample.position - ray_position).length() - 2.f*RAY_OFFSET);

            SurfaceSample& sample = samples.push();

            sample.intersects = resources.intersector->intersect(ray, &next_path, resources, 
                transmit ? transmit_interiors.get_callback() : interiors.get_callback());

            Transmittance transmittance = Integrator::shadow(sample.intersects, resources);
            Vec3f shadow = transmittance.shadow(ray.tmax, resources);

            sample.li = shadow * light_sample.intensity;
            sample.weight = weight;
            sample.pdf = light_sample.pdf;

            Vec3f color = (sample.li * sample.weight) / sample.pdf;
            if(num_samples >= min_num_samples && luminance(variance_sum / (num_samples-1.f) - (variance_sum+color) / num_samples) < 0.01f)
                break;
            variance_sum += color;
        }

        float inv_num_samples = 1.f / num_samples;
        for(uint i = samples_begin; i < samples.size(); i++)
            samples[i].weight *= inv_num_samples;
    }
}

float SurfaceLightSampler::evaluate(
    const SurfaceSample& sample, const MaterialLobes& lobes,
    const Intersect * intersect, SurfaceSamplerData * data, 
    Resources& resources) const
{
    if(sample.scatter)
        return false;

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

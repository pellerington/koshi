#include <koshi/integrator/SurfaceSampler.h>

#include <koshi/intersection/Intersector.h>
#include <koshi/integrator/Transmittance.h>
#include <koshi/light/LightSampler.h>
#include <koshi/base/Scene.h>

void SurfaceSampler::pre_render(Resources& resources)
{
    min_quality = std::pow(1.f / SAMPLES_PER_HEMISPHERE, resources.settings->depth);

    uint i = 0;
    for(auto object = resources.scene->begin(); object != resources.scene->end(); ++object)
    {
        Geometry * geometry = dynamic_cast<Geometry*>(object->second);
        if(geometry)
        {
            LightSampler * sampler = geometry->get_attribute<LightSampler>("light_sampler");
            if(sampler) 
            {
                lights[geometry].sampler = sampler;
                lights[geometry].id = i++;
            }
        }
    }
}

void * SurfaceSampler::pre_integrate(const Intersect * intersect, Resources& resources)
{
    SurfaceSamplerData * data = resources.memory->create<SurfaceSamplerData>();
    data->surface = dynamic_cast<const Surface *>(intersect->geometry_data);
    return data;
}

Vec3f SurfaceSampler::integrate(const Intersect * intersect, void * data, Transmittance& transmittance, Resources& resources) const
{
    // Check it is geometry surface.
    SurfaceSamplerData * surface_data = (SurfaceSamplerData *)data;
    const Surface * surface = surface_data->surface;
    if(!surface || !surface->material) return VEC3F_ZERO;

    // Setup color and shadow variables.
    Vec3f color = VEC3F_ZERO;
    Vec3f shadow = transmittance.shadow(intersect->t, resources) * surface->opacity;

    // Add light contribution.
    color += (surface->facing) ? surface->material->emission(surface->u, surface->v, surface->w, intersect, resources) : VEC3F_ZERO;

    // Terminate before scattering.
    const uint depth = intersect->path ? intersect->path->depth : 0;
    const float quality = intersect->path ? intersect->path->quality : 1.f;
    if(is_saturated(color) || depth > resources.settings->max_depth || quality < min_quality || shadow.null())
        return color * shadow;

    // Setup material data.
    MaterialLobes lobes = surface->material->instance(surface, intersect, resources);
    if(!lobes.size()) 
        return color * shadow;

    // Setup light data.
    Array<const LightSamplerData *> light_data(resources.memory, lights.size());
    for(uint i = 0; i < lights.size(); i++)
        light_data.push(nullptr);

    // Set number of samples.
    uint max_samples = SAMPLES_PER_HEMISPHERE * quality * resources.settings->sampling_quality;
    float desired_samples = 0;
    //if material_sample
    {
        for(uint l = 0; l < lobes.size(); l++)
        {
            const MaterialLobe * lobe = lobes[l];
            MaterialLobe::ScatterType scatter_type = lobe->get_scatter_type();
            if(scatter_type == MaterialLobe::DIFFUSE || scatter_type == MaterialLobe::SUBSURFACE)
                desired_samples += SAMPLES_PER_HEMISPHERE;
            else if(scatter_type == MaterialLobe::GLOSSY)
                desired_samples += SAMPLES_PER_HEMISPHERE * sqrtf(lobe->roughness);
            else
                desired_samples += 1.f;
        }
    }
    //if light_sample
    {
        for(auto light = lights.begin(); light != lights.end(); ++light)
        {
            LightSampler * sampler = light->second.sampler;
            LightSampler::LightType light_type = sampler->get_light_type();
            desired_samples += (light_type == LightSampler::POINT) ? 1.f : SAMPLES_PER_HEMISPHERE;
        }
    }

    // Material Sampling
    for(uint l = 0; l < lobes.size(); l++)
    {
        const MaterialLobe * lobe = lobes[l];

        MaterialLobe::ScatterType scatter_type = lobe->get_scatter_type();
        MaterialLobe::Hemisphere hemisphere = lobe->get_hemisphere();

        float num_samples = 1.f;
        if(scatter_type == MaterialLobe::DIFFUSE || scatter_type == MaterialLobe::SUBSURFACE)
            num_samples = SAMPLES_PER_HEMISPHERE;
        else if(scatter_type == MaterialLobe::GLOSSY)
            num_samples = SAMPLES_PER_HEMISPHERE * sqrtf(lobe->roughness);

        // TODO: Why attach recursion data (nicer name) to the intersect...
        PathData next_path;
        next_path.depth = depth + 1;
        next_path.quality = quality / num_samples;
        next_path.prev_path = intersect->path;

        num_samples = std::max(1.f, max_samples * num_samples / desired_samples);
        const float inv_num_samples = 1.f / num_samples;

        for(uint i = 0; i < num_samples; i++)
        {
            MaterialSample sample;
            if(!lobe->sample(sample, resources))
                continue;

            // TODO: This method doesnt include any of the information we need to scatter like transmittance ect...
            // AND it's ugly... scatter method ???
            if(scatter_type == MaterialLobe::SUBSURFACE)
            {
                color += sample.value * inv_num_samples;
                continue;
            }

            const bool front = sample.wo.dot(surface->normal) > 0.f;
            const bool transmit = front ^ surface->facing;

            if((transmit && hemisphere == MaterialLobe::FRONT) || (!transmit && hemisphere == MaterialLobe::BACK))
                continue;

            const Vec3f& ray_position = front ? surface->front_position : surface->back_position;
            Ray ray(ray_position, sample.wo);
            IntersectionCallbacks * interior_callback = (transmit) ? transmittance.get_interiors_callback(intersect->t, surface->facing ? Transmittance::PUSH : Transmittance::POP, intersect->geometry, lobe->interior) : nullptr;
            IntersectList * intersects = resources.intersector->intersect(ray, &next_path, resources, interior_callback);

            Vec3f li = Integrator::shade(intersects, resources);

            // if(light_sampling)
            {
                float light_pdf = 0.f;
                for(uint i = 0; i < intersects->size(); i++)
                {
                    const Intersect * light_intersect = intersects->get(i);
                    auto light = lights.find(light_intersect->geometry);
                    if(light == lights.end()) continue;

                    LightSampler * sampler = light->second.sampler;
                    const uint& id = light->second.id;
                    const LightSamplerData * sampler_data = light_data[id];
                    if(!sampler_data)
                        sampler_data = light_data[id] = sampler->pre_integrate(surface, resources);

                    light_pdf += sampler->evaluate(light_intersect, sampler_data, resources);
                }

                sample.value *= (sample.pdf * sample.pdf) / (sample.pdf * sample.pdf + light_pdf * light_pdf);
            }

            color += li * sample.value * inv_num_samples / sample.pdf;
        }
    }

    // Light Sampling
    for(auto light = lights.begin(); light != lights.end(); ++light)
    {
        LightSampler * sampler = light->second.sampler;
        const uint& id = light->second.id;
        const LightSamplerData * sampler_data = light_data[id];
        if(!sampler_data)
            sampler_data = light_data[id] = sampler->pre_integrate(surface, resources);

        LightSampler::LightType light_type = sampler->get_light_type();
        float num_samples = 1.f;
        if(light_type == LightSampler::AREA)
            num_samples = SAMPLES_PER_HEMISPHERE;

        PathData next_path;
        next_path.depth = depth + 1;
        next_path.quality = quality / num_samples;
        next_path.prev_path = intersect->path;

        num_samples = std::max(1.f, max_samples * num_samples / desired_samples);
        const float inv_num_samples = 1.f / num_samples;

        for(uint i = 0; i < num_samples; i++)
        {
            LightSample sample;
            if(!sampler->sample(sample, sampler_data, resources))
                continue;

            Vec3f wo = (sample.position - surface->position).normalized();
            const bool front = wo.dot(surface->normal) > 0.f;
            const bool transmit = front ^ surface->facing;

            Vec3f weight;
            float material_pdf = 0.f;
            MaterialSample material_sample;
            material_sample.wo = wo;
            for(size_t i = 0; i < lobes.size(); i++)
            {
                MaterialLobe::Hemisphere hemisphere = lobes[i]->get_hemisphere();
                if((hemisphere == MaterialLobe::SPHERE 
                || (!transmit && hemisphere == MaterialLobe::FRONT) 
                || (transmit && hemisphere == MaterialLobe::BACK))
                && lobes[i]->evaluate(material_sample, resources))
                {
                    weight += material_sample.value;
                    material_pdf += material_sample.pdf;
                }
            }

            // if(material_sample)
                weight *= (sample.pdf * sample.pdf) / (sample.pdf * sample.pdf + material_pdf * material_pdf);

            if(is_black(weight)) continue;

            const Vec3f& ray_position = front ? surface->front_position : surface->back_position;
            Ray ray(ray_position, wo, 0.f, (sample.position - ray_position).length() - 2.f*RAY_OFFSET);
            IntersectionCallbacks * interior_callback = (!surface->facing && transmit) ? transmittance.get_interiors_callback(intersect->t, Transmittance::POP, intersect->geometry) : nullptr;
            IntersectList * intersects = resources.intersector->intersect(ray, &next_path, resources, interior_callback);

            Transmittance transmittance = Integrator::shadow(intersects, resources);
            Vec3f shadow = transmittance.shadow(ray.tmax, resources);

            color += shadow * sample.intensity * weight * inv_num_samples / sample.pdf;
        }
    }

    return color * shadow;
}

Vec3f SurfaceSampler::shadow(const float& t, const Intersect * intersect, void * data, Resources& resources) const
{
    SurfaceSamplerData * surface_data = (SurfaceSamplerData *)data;
    if(!surface_data->surface) return VEC3F_ONES;
    return (t > intersect->t) ? (VEC3F_ONES - surface_data->surface->opacity) : VEC3F_ONES;
}
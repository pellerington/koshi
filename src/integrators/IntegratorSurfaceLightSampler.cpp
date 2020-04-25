#include <integrators/IntegratorSurfaceLightSampler.h>
#include <Scene/Scene.h>

void IntegratorSurfaceLightSampler::pre_render(Scene * scene)
{
    std::vector<Object*>& objects = scene->get_objects();
    for(size_t i = 0; i < objects.size(); i++)
    {
        Geometry * geometry = dynamic_cast<Geometry*>(objects[i]);
        if(geometry)
        {
            LightSampler * sampler = geometry->get_attribute<LightSampler>("light_sampler");
            if(sampler) lights[geometry] = sampler;
        }
    }
}

std::vector<SurfaceSample> IntegratorSurfaceLightSampler::integrate_surface(
    MaterialInstance * material_instance, Material * material, 
    const Intersect& intersect, const GeometrySurface * surface, 
    Resources& resources) const
{
    const PathData * prev_path = intersect.path;
    const uint depth = prev_path ? prev_path->depth : 0;
    const float quality = prev_path ? prev_path->quality : 1.f;

    std::vector<SurfaceSample> samples;
    // if(resources.settings->sample_lights)
    for(auto light = lights.begin(); light != lights.end(); ++light)
    {
        // Make a better num_samples estimator
        uint num_samples = std::max(1.f, SAMPLES_PER_SA * quality * resources.settings->sampling_quality);
        std::vector<LightSample> light_samples;
        light->second->sample_light(num_samples, intersect, light_samples, resources);
        num_samples = light_samples.size();
        const float inv_num_samples = 1.f / (float)num_samples;
        for(uint i = 0; i < num_samples; i++)
        {
            PathData next_path;
            next_path.depth = depth + 1;
            next_path.quality = quality / num_samples;
            next_path.prev_path = prev_path;

            Vec3f wo = (light_samples[i].position - surface->position).normalized();
            MaterialSample material_sample;
            material_sample.wo = wo;
            if(!material->evaluate_material(material_instance, material_sample)) continue;

            const bool reflect = wo.dot(intersect.surface.normal) > 0;
            const Vec3f ray_pos = reflect ? intersect.surface.front_position : intersect.surface.back_position;
            Ray ray(ray_pos, wo, 0.f, (light_samples[i].position - ray_pos).length() - EPSILON_F);
            samples.emplace_back(resources.intersector->intersect(ray, &next_path));
            
            // samples.back().intersects = resources.intersector->intersect(ray, &next_path);

            // Replace this with transmittance context later
            Vec3f shadow = 1.f;
            if(samples.back().intersects.hit())
                shadow = 0.f;

            samples.back().li = shadow * light_samples[i].intensity;
            samples.back().material_sample = material_sample;
            samples.back().weight = inv_num_samples;
            samples.back().pdf = light_samples[i].pdf;
        }
    }

    return samples;
}

float IntegratorSurfaceLightSampler::evaluate(const Intersect& intersect, const SurfaceSample& sample, Resources& resources)
{
    float pdf = 0.f;
    for(uint i = 0; i < sample.intersects.size(); i++)
    {
        auto light = lights.find(sample.intersects[i].geometry);
        if(light == lights.end()) continue;
        pdf += light->second->evaluate_light(sample.intersects[i], intersect, resources);
    }
    return pdf;
}

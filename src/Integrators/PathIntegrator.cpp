#include <Integrators/PathIntegrator.h>

#include <cmath>
#include <Util/Color.h>
#include <Volume/ZeroScatVolumeIntegrator.h>
#include <Volume/MultiScatVolumeIntegrator.h>

#include <intersection/Intersect.h>

void PathIntegrator::pre_render()
{
    quality_threshold = std::pow(1.f / SAMPLES_PER_SA, scene->settings.depth);
}

Vec3f PathIntegrator::integrate(Ray& ray, PathSample& in_sample, Resources &resources) const
{
    // Initialize output
    Vec3f color = VEC3F_ZERO;

    // Intersect with our scene
    const IntersectList intersects = resources.intersector->intersect(ray);

    // If this is a light sample shadow and return the value
    if(in_sample.type == PathSample::Light)
    {
        // ZeroScatVolumeIntegrator volume_integrator(scene, ray, intersect.volumes, resources);
        return (intersects.hit()) ? Vec3f(0.f) : /*volume_integrator.shadow(ray.tmax) * */ in_sample.lsample->intensity;
    }

    const Intersect& intersect = intersects[0];

    if(!intersect.geometry)
        return color;

    LightSampler * light_sampler = intersect.geometry->get_attribute<LightSampler>("light_sampler");
    if(light_sampler && in_sample.type != PathSample::Camera)
    {
        light_sampler->evaluate_light(intersect, *in_sample.intersect, *in_sample.lsample, resources);
        color += /*shadow_tmax * */ in_sample.lsample->intensity;
    }
    else if(intersect.geometry->light)
    {
        color += intersect.geometry->light->get_intensity(intersect, resources);
    }   

    // Check if we should terminate early.
    if(is_saturated(color) || in_sample.depth > scene->settings.max_depth)
        return color;

    //Kill our ray if we are below the threshold.
    if(in_sample.quality < quality_threshold)
        return color;

    // Scatter the surface
    color +=  /*shadow_tmax * */scatter_surface(intersect, in_sample, resources);

    return color;
}

Vec3f PathIntegrator::scatter_surface(const Intersect &intersect, PathSample &in_sample, Resources &resources) const
{
    // Initialize output
    Vec3f color = VEC3F_ZERO;

    // Terminate if we have no material.
    std::shared_ptr<Material> material = intersect.geometry->material;
    if(!material) return color;
    
    // Get our material instance
    MaterialInstance * material_instance = material->instance(&intersect.surface, resources);
    const float sample_multiplier = scene->settings.quality * in_sample.quality;

    // Sample the material
    std::vector<MaterialSample> material_samples;
    if(scene->settings.sample_material)
        material->sample_material(material_instance, material_samples, sample_multiplier, resources);
    const uint n_material_samples = material_samples.size();

    // Sample the lights
    std::vector<LightSample> light_samples;
    if(scene->settings.sample_lights)
        scene->sample_lights(intersect, light_samples, sample_multiplier, resources);
    const uint n_light_samples = light_samples.size();

    // Return if we have no samples
    if(!(n_material_samples + n_light_samples))
        return color;

    const bool multiple_importance_sample = (n_light_samples > 0) && (n_material_samples > 0);
    const float material_sample_weight = (n_material_samples > 0) ? 1.f / n_material_samples : 0.f;
    const float light_sample_weight = (n_light_samples > 0) ? 1.f / n_light_samples : 0.f;

    // Perform the integrations
    for(uint i = 0; i < material_samples.size(); i++)
    {
        PathSample sample;
        sample.parent = &in_sample;
        sample.intersect = &intersect;
        sample.depth = in_sample.depth + 1;
        sample.quality = in_sample.quality * material_samples[i].quality;
        sample.msample = &material_samples[i];
        const bool reflect = sample.msample->wo.dot(intersect.surface.normal) > 0;
        sample.type = reflect ? PathSample::Reflection : PathSample::Transmission;
        LightSample lsample;
        sample.lsample = &lsample;

        Ray ray(reflect ? intersect.surface.front_position : intersect.surface.back_position, sample.msample->wo);
        // ray.in_volumes = reflect ? intersect.volumes.get_passthrough_volumes() : intersect.volumes.get_passthrough_transmission_volumes();

        const Vec3f in_color = (is_black(sample.msample->weight)) ? 0.f : integrate(ray, sample, resources);

        float weight = material_sample_weight;
        if(multiple_importance_sample)
            weight *= (sample.msample->pdf * sample.msample->pdf) / ((sample.msample->pdf * sample.msample->pdf) + (sample.lsample->pdf * sample.lsample->pdf));

        color += in_color * sample.msample->weight * weight / sample.msample->pdf;
    }

    for(uint i = 0; i < light_samples.size(); i++)
    {
        PathSample sample;
        sample.parent = &in_sample;
        sample.intersect = &intersect;
        sample.depth = in_sample.depth + 1;
        sample.quality = 1.f;
        sample.lsample = &light_samples[i];
        sample.type = PathSample::Light;
        MaterialSample msample;
        sample.msample = &msample;
        msample.wo = (sample.lsample->position - intersect.surface.position).normalized();

        if(!material->evaluate_material(material_instance, msample)) continue;

        const float wo_dot_n = sample.msample->wo.dot(intersect.surface.normal);
        Ray ray((wo_dot_n >= 0.f) ? intersect.surface.front_position : intersect.surface.back_position, 
                sample.msample->wo, 0.f, (sample.lsample->position - ray.pos).length() - EPSILON_F);

        const Vec3f in_color = integrate(ray, sample, resources);

        float weight = light_sample_weight;
        if(multiple_importance_sample)
            weight *= (sample.lsample->pdf * sample.lsample->pdf) / ((sample.msample->pdf * sample.msample->pdf) + (sample.lsample->pdf * sample.lsample->pdf));

        color += in_color * sample.msample->weight * weight / sample.lsample->pdf;
    }

    return color;
}

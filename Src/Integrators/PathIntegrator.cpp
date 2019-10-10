#include "PathIntegrator.h"

#include <cmath>
#include "../Util/Color.h"
#include "../Volume/VolumeIntegrator.h"

void PathIntegrator::pre_render()
{
}

Vec3f PathIntegrator::integrate(Ray &ray, PathSample &in_sample) const
{
    // Initialize output
    Vec3f color = VEC3F_ZERO;

    // If we have a light sample set our tmax
    if(in_sample.type == PathSample::Light)
        ray.t = (in_sample.lsample->position - ray.pos).length();

    // Intersect with our scene
    VolumeStack volumes(in_sample.volumes);
    Surface surface = scene->intersect(ray, &volumes);

    // Volume stack and integrator should be one thing.
    VolumeIntegrator volume_integrator(volumes);

    // If we are unshadowed return the light sample intensity
    if(in_sample.type == PathSample::Light)
        return (ray.hit) ? Vec3f(0.f) : volume_integrator.shadow(ray.t) * in_sample.lsample->intensity;

    // Check if we intersected any lights
    std::vector<LightSample> light_results;
    if(in_sample.type != PathSample::Camera || scene->settings.display_lights)
        scene->evaluate_lights(ray, light_results);
    for(uint i = 0; i < light_results.size(); i++)
    {
        const float dist = (light_results[i].position - ray.pos).length();
        color += volume_integrator.shadow(dist) * light_results[i].intensity;
        if(in_sample.lsample)
        {
            in_sample.lsample->intensity += light_results[i].intensity;
            in_sample.lsample->pdf += light_results[i].pdf;
        }
    }

    // If we hit nothing end it here. Incorporate this into our main evaluate light.
    if(!ray.hit)
    {
        Vec3f environment = scene->evaluate_environment_light(ray);
        if(in_sample.type != PathSample::Camera || scene->settings.display_lights)
            color += environment;
        return color;
    }

    // Check if we should terminate.
    if(in_sample.depth > scene->settings.max_depth || is_saturated(color)) //Checking saturation may be incorrect.
        return color;

    // Volume in-scattering goes here

    // Integrate the surface
    color += volume_integrator.shadow(ray.t) * integrate_surface(surface, in_sample);

    return color;
}

Vec3f PathIntegrator::integrate_surface(const Surface &surface, PathSample &in_sample) const
{
    // Initialize output
    Vec3f color = VEC3F_ZERO;

    // Get our material instance
    std::shared_ptr<Material> material = (surface.object->material) ? surface.object->material->instance(&surface) : std::shared_ptr<Material>(new Material());

    // Add any emissive component of the material. If our color is fully saturated end here.
    color += material->get_emission();
    if(is_saturated(color)) //Checking saturation may be incorrect.
        return color;

    // Setup sampling variables
    const float sample_multiplier = scene->settings.quality * in_sample.quality;

    // Sample the material
    std::vector<MaterialSample> material_samples;
    if(scene->settings.sample_material)
        material->sample_material(material_samples, sample_multiplier);
    const uint n_material_samples = material_samples.size();

    // Sample the lights
    std::vector<LightSample> light_samples;
    if(scene->settings.sample_lights)
        scene->sample_lights(surface, light_samples, sample_multiplier);
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
        sample.depth = in_sample.depth + 1;
        sample.volumes = surface.volumes;
        sample.quality = in_sample.quality * material_samples[i].quality;
        sample.msample = &material_samples[i];
        sample.type = PathSample::Material;
        LightSample lsample;
        sample.lsample = &lsample;

        const float wo_dot_n = sample.msample->wo.dot(surface.normal);
        Ray ray((wo_dot_n >= 0.f) ? surface.front_position : surface.back_position, sample.msample->wo);

        Vec3f in_color = integrate(ray, sample);

        float weight = material_sample_weight;
        if(multiple_importance_sample)
            weight *= (sample.msample->pdf * sample.msample->pdf) / ((sample.msample->pdf * sample.msample->pdf) + (sample.lsample->pdf * sample.lsample->pdf));

        color += in_color * sample.msample->fr * weight / sample.msample->pdf;
    }

    for(uint i = 0; i < light_samples.size(); i++)
    {
        PathSample sample;
        sample.parent = &in_sample;
        sample.volumes = surface.volumes;
        sample.depth = in_sample.depth + 1;
        sample.quality = 1.f;
        sample.lsample = &light_samples[i];
        sample.type = PathSample::Light;
        MaterialSample msample;
        sample.msample = &msample;
        msample.wo = (sample.lsample->position - surface.position).normalized();

        if(!material->evaluate_material(msample)) continue;

        const float wo_dot_n = sample.msample->wo.dot(surface.normal);
        Ray ray((wo_dot_n >= 0.f) ? surface.front_position : surface.back_position, sample.msample->wo);

        Vec3f in_color = integrate(ray, sample);

        float weight = light_sample_weight;
        if(multiple_importance_sample)
            weight *= (sample.lsample->pdf * sample.lsample->pdf) / ((sample.msample->pdf * sample.msample->pdf) + (sample.lsample->pdf * sample.lsample->pdf));

        color += in_color * sample.msample->fr * weight / sample.lsample->pdf;
    }

    return color;
}

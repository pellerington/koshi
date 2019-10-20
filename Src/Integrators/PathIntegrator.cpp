#include "PathIntegrator.h"

#include <cmath>
#include "../Util/Color.h"
#include "../Volume/ZeroScatVolumeIntegrator.h"
#include "../Volume/MultiScatVolumeIntegrator.h"

#include "../Util/Intersect.h"

void PathIntegrator::pre_render()
{
}

Vec3f PathIntegrator::integrate(Ray &ray, PathSample &in_sample) const
{
    // Initialize output
    Vec3f color = VEC3F_ZERO;

    // Intersect with our scene
    Intersect intersect = scene->intersect(ray);


    // If this is a light sample shadow and return the value
    if(in_sample.type == PathSample::Light)
    {
        ZeroScatVolumeIntegrator volume_integrator(scene, ray, intersect.volumes);
        return (ray.hit) ? Vec3f(0.f) : volume_integrator.shadow(ray.tmax) * in_sample.lsample->intensity;
    }

    // Create a volume integrator
    MultiScatVolumeIntegrator volume_integrator(scene, ray, intersect.volumes /*Add volume insample here*/);

    // Add the emission from our volume
    color += volume_integrator.emission();

    // If we hit nothing end it here. Incorporate this into our main evaluate light?
    if(!ray.hit && !intersect.volumes.num_volumes() /* <- replace this with no scattering? */)
    {
        Vec3f environment = scene->evaluate_environment_light(ray);
        if(in_sample.type != PathSample::Camera || scene->settings.display_lights)
            color += environment;
        return color;
    }

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

    // Check if we should terminate.
    if(is_saturated(color) || in_sample.depth > scene->settings.max_depth)
        return color;

    // TODO: this is causing problems
    // Exit if our sample is too low quality
    const float kill_prob = std::min(1.f, in_sample.quality * 16.f /* <- Let a user set this */);
    if(RNG::Rand() > kill_prob)
        return color / kill_prob;

    // TODO: change this to a scatter points so that we can sample lights from them!
    // Scatter our volume
    std::vector<VolumeSample> volume_samples;
    volume_integrator.scatter(volume_samples);
    for(uint i = 0; i < volume_samples.size(); i++)
    {
        PathSample sample;
        sample.depth = in_sample.depth + 1;
        sample.quality = in_sample.quality * volume_samples[i].quality;
        sample.vsample = &volume_samples[i];
        sample.type = PathSample::Volume;
        LightSample lsample;
        sample.lsample = &lsample;

        Ray ray(sample.vsample->pos, sample.vsample->wo);
        ray.in_volumes = sample.vsample->exit_volumes;

        color += sample.vsample->weight * integrate(ray, sample);
    }

    // Integrate the surface
    if(ray.hit)
    {
        const Vec3f shadow = volume_integrator.shadow(ray.t);
        color += !is_black(shadow) ? shadow * integrate_surface(intersect, in_sample) : VEC3F_ZERO;
    }

    return color / kill_prob;
}

Vec3f PathIntegrator::integrate_surface(const Intersect &intersect, PathSample &in_sample) const
{
    // Initialize output
    Vec3f color = VEC3F_ZERO;

    // Get our material instance
    std::shared_ptr<Material> material = (intersect.object->material) ? intersect.object->material->instance(&intersect.surface)
                                                                      : std::shared_ptr<Material>(new Material());

    // Add any emissive component of the material. If our color is fully saturated end here.
    color += material->get_emission();
    if(is_saturated(color))
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
        scene->sample_lights(intersect.surface, light_samples, sample_multiplier);
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
        sample.quality = in_sample.quality * material_samples[i].quality;
        sample.msample = &material_samples[i];
        sample.type = PathSample::Material;
        LightSample lsample;
        sample.lsample = &lsample;

        const float wo_dot_n = sample.msample->wo.dot(intersect.surface.normal);
        Ray ray((wo_dot_n >= 0.f) ? intersect.surface.front_position : intersect.surface.back_position, sample.msample->wo);
        ray.in_volumes = intersect.volumes.get_exit_volumes();

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
        sample.depth = in_sample.depth + 1;
        sample.quality = 1.f;
        sample.lsample = &light_samples[i];
        sample.type = PathSample::Light;
        MaterialSample msample;
        sample.msample = &msample;
        const Vec3f dir = sample.lsample->position - intersect.surface.position;
        msample.wo = dir.normalized();

        if(!material->evaluate_material(msample)) continue;

        const float wo_dot_n = sample.msample->wo.dot(intersect.surface.normal);
        Ray ray((wo_dot_n >= 0.f) ? intersect.surface.front_position : intersect.surface.back_position, sample.msample->wo);
        ray.in_volumes = intersect.volumes.get_exit_volumes();
        ray.tmax = dir.length(); // Set our tmax so we can perform shadowing.

        Vec3f in_color = integrate(ray, sample);

        float weight = light_sample_weight;
        if(multiple_importance_sample)
            weight *= (sample.lsample->pdf * sample.lsample->pdf) / ((sample.msample->pdf * sample.msample->pdf) + (sample.lsample->pdf * sample.lsample->pdf));

        color += in_color * sample.msample->fr * weight / sample.lsample->pdf;
    }

    return color;
}

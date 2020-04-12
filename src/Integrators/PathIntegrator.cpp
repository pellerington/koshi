#include "PathIntegrator.h"

#include <cmath>
#include "../Util/Color.h"
#include "../Volume/ZeroScatVolumeIntegrator.h"
#include "../Volume/MultiScatVolumeIntegrator.h"

#include "../Util/Intersect.h"

void PathIntegrator::pre_render()
{
    quality_threshold = std::pow(1.f / SAMPLES_PER_SA, scene->settings.depth);
}

Vec3f PathIntegrator::integrate(Ray &ray, PathSample &in_sample, Resources &resources) const
{
    // Initialize output
    Vec3f color = VEC3F_ZERO;

    // Intersect with our scene
    const Intersect intersect = scene->intersect(ray);

    // If this is a light sample shadow and return the value
    if(in_sample.type == PathSample::Light)
    {
        ZeroScatVolumeIntegrator volume_integrator(scene, ray, intersect.volumes, resources);
        return (ray.hit) ? Vec3f(0.f) : volume_integrator.shadow(ray.tmax) * in_sample.lsample->intensity;
    }

    // Create a volume integrator (This should use scratchpad memory)
    std::unique_ptr<VolumeIntegrator> volume_integrator = (in_sample.quality == 1.f)
        ? std::unique_ptr<VolumeIntegrator>( new MultiScatVolumeIntegrator(scene, ray, intersect.volumes, dynamic_cast<VolumeSample*>(in_sample.msample), resources))
        : std::unique_ptr<VolumeIntegrator>( new ZeroScatVolumeIntegrator(scene, ray, intersect.volumes, resources));

    // Add the emission from our volume
    color += volume_integrator->emission();

    // Evaluate any light we hit.
    const Vec3f shadow_tmax = volume_integrator->shadow(ray.t);
    if(!is_black(shadow_tmax) && intersect.object)
    {
        if(intersect.object->evaluate_light(intersect.surface, &ray.pos, nullptr, *in_sample.lsample, resources))
            color += shadow_tmax * in_sample.lsample->intensity;
    }

    // Check if we should terminate early.
    if(is_saturated(color) || in_sample.depth > scene->settings.max_depth || (!ray.hit && !intersect.volumes.num_volumes()))
        return color;

    //Kill our ray if we are below the threshold.
    if(in_sample.quality < quality_threshold)
        return color;

    // Scatter our volume
    std::vector<VolumeSample> volume_samples;
    volume_integrator->scatter(volume_samples);
    for(uint i = 0; i < volume_samples.size(); i++)
    {
        PathSample sample;
        sample.depth = in_sample.depth + 1;
        sample.quality = in_sample.quality * volume_samples[i].quality;
        sample.msample = &volume_samples[i];
        sample.type = PathSample::Volume;
        LightSample lsample;
        sample.lsample = &lsample;

        Ray ray(volume_samples[i].pos, volume_samples[i].wo);
        ray.ior = intersect.ior;
        ray.in_volumes = volume_samples[i].passthrough_volumes;

        color += sample.msample->weight * integrate(ray, sample, resources);
    }

    // Scatter the surface
    if(!is_black(shadow_tmax) && ray.hit)
        color +=  shadow_tmax * scatter_surface(intersect, in_sample, resources);

    return color;
}

Vec3f PathIntegrator::scatter_surface(const Intersect &intersect, PathSample &in_sample, Resources &resources) const
{
    // Initialize output
    Vec3f color = VEC3F_ZERO;

    // Terminate if we have no material.
    std::shared_ptr<Material> material = intersect.object->material;
    if(!material) return color;

    // Get our material instance
    MaterialInstance * material_instance = material->instance(&intersect.surface, resources);
    const float sample_multiplier = scene->settings.quality * in_sample.quality;
    IorStack transmission_ior = intersect.surface.front ? IorStack(material->get_ior(/* materialinstance */), intersect.ior) : *intersect.ior->get_prev_stack();

    // Sample the material
    std::vector<MaterialSample> material_samples;
    if(scene->settings.sample_material)
        material->sample_material(material_instance, material_samples, sample_multiplier, resources);
    const uint n_material_samples = material_samples.size();

    // Sample the lights
    std::vector<LightSample> light_samples;
    if(scene->settings.sample_lights)
        scene->sample_lights(intersect.surface, light_samples, sample_multiplier, resources);
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
        const bool reflect = sample.msample->wo.dot(intersect.surface.normal) > 0;
        sample.type = reflect ? PathSample::Reflection : PathSample::Transmission;
        LightSample lsample;
        sample.lsample = &lsample;

        Ray ray(reflect ? intersect.surface.front_position : intersect.surface.back_position, sample.msample->wo);
        ray.ior = reflect ? intersect.ior : &transmission_ior;
        ray.in_volumes = reflect ? intersect.volumes.get_passthrough_volumes() : intersect.volumes.get_passthrough_transmission_volumes();

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
        sample.depth = in_sample.depth + 1;
        sample.quality = 1.f;
        sample.lsample = &light_samples[i];
        sample.type = PathSample::Light;
        MaterialSample msample;
        sample.msample = &msample;
        msample.wo = (sample.lsample->position - intersect.surface.position).normalized();

        if(!material->evaluate_material(material_instance, msample)) continue;

        const float wo_dot_n = sample.msample->wo.dot(intersect.surface.normal);
        Ray ray((wo_dot_n >= 0.f) ? intersect.surface.front_position : intersect.surface.back_position, sample.msample->wo);
        ray.ior = intersect.ior;
        ray.in_volumes = (wo_dot_n >= 0.f) ? intersect.volumes.get_passthrough_volumes() : intersect.volumes.get_passthrough_transmission_volumes();
        ray.tmax = (sample.lsample->position - ray.pos).length() - EPSILON_F; // Set our tmax so we can perform shadowing.

        const Vec3f in_color = integrate(ray, sample, resources);

        float weight = light_sample_weight;
        if(multiple_importance_sample)
            weight *= (sample.lsample->pdf * sample.lsample->pdf) / ((sample.msample->pdf * sample.msample->pdf) + (sample.lsample->pdf * sample.lsample->pdf));

        color += in_color * sample.msample->weight * weight / sample.lsample->pdf;
    }

    return color;
}

#include "PathIntegrator.h"

#include <cmath>

#include "../Util/Color.h"


void PathIntegrator::pre_render()
{
}

Vec3f PathIntegrator::integrate(Ray &ray, const float current_quality, PathSample * in_sample) const
{
    // Initialize output
    Vec3f color = VEC3F_ZERO;
    Vec3f emission = VEC3F_ZERO;

    // If we have a light sample set our tmax
    // In future override this with a explicit shadowing function
    if(in_sample && in_sample->type == PathSample::Light)
        ray.t = (in_sample->lsample->position - ray.pos).length();

    // Intersect with our scene
    Surface surface;
    scene->intersect(ray, surface);

    // If we are unshadowed return the light sample intensity
    // In future override this with a explicit shadowing function
    if(in_sample && in_sample->type == PathSample::Light)
        return (ray.hit) ? Vec3f(0.f) : in_sample->lsample->intensity;

    // Check if we intersected any lights
    LightSample light_result;
    if(ray.depth > 0 || scene->settings.display_lights)
        scene->evaluate_lights(ray, light_result);
    emission += light_result.intensity;
    if(in_sample) *(in_sample->lsample) = light_result;

    // If we hit nothing end it here
    // Incorporate this into our main evaluate light!
    if(!ray.hit && ray.t == FLT_MAX)
    {
        Vec3f environment = scene->evaluate_environment_light(ray);
        if(ray.depth > 0 || scene->settings.display_lights)
            emission += environment;
        return emission;
    }

    // Volume Function goes here
    // Intersect some volume structure.
    // Sample points/directions along it.
    // Sucess?

    // Move this to a surface function

    // if(!material) Use some default material.
    std::shared_ptr<Material> material = surface.object->material->instance(surface);

    // Add any emissive component of the material to the integrator
    emission += material->get_emission();

    // If we are greater than the max depth, or our emission is fully saturated end here.
    if(ray.depth > scene->settings.max_depth || is_saturated(emission)) // Saturated emission may not be correct?
        return emission;

    // Setup sampling variables
    const float sample_multiplier = scene->settings.quality * current_quality;

    // Sample the material
    std::deque<MaterialSample> material_samples;
    if(scene->settings.sample_material)
        material->sample_material(surface, material_samples, sample_multiplier);
    const uint n_material_samples = material_samples.size();

    // Sample the lights
    std::deque<LightSample> light_samples;
    if(scene->settings.sample_lights)
        scene->sample_lights(surface, light_samples, sample_multiplier);
    const uint n_light_samples = light_samples.size();

    // Return if we have no samples
    if(!(n_material_samples + n_light_samples))
        return emission;

    const bool multiple_importance_sample = (n_light_samples > 0) && (n_material_samples > 0);
    const float material_sample_weight = (n_material_samples > 0) ? 1.f / n_material_samples : 0.f;
    const float light_sample_weight = (n_light_samples > 0) ? 1.f / n_light_samples : 0.f;

    // These should be on the surface object!!!
    //surface.position + ((wo_dot_n >= 0.f) ? (surface.normal * EPSILON_F) : (surface.normal * -EPSILON_F)); // Get this from surface
    const Vec3f front_position = surface.position + surface.normal *  EPSILON_F;
    const Vec3f back_position  = surface.position + surface.normal * -EPSILON_F;

    for(uint i = 0; i < material_samples.size(); i++)
    {
        PathSample sample;
        sample.msample = &material_samples[i];
        sample.type = PathSample::Material;
        LightSample lsample;
        sample.lsample = &lsample;

        const float wo_dot_n = sample.msample->wo.dot(surface.normal);
        Ray sample_ray((wo_dot_n >= 0.f) ? front_position : back_position, sample.msample->wo);
        sample_ray.depth = ray.depth + 1; // Depth should be in the path_sample

        Vec3f in_color = integrate(sample_ray, current_quality * sample.msample->quality, &sample);

        float weight = material_sample_weight;
        if(multiple_importance_sample)
            weight *= (sample.msample->pdf * sample.msample->pdf) / ((sample.msample->pdf * sample.msample->pdf) + (sample.lsample->pdf * sample.lsample->pdf));

        color += in_color * sample.msample->fr * weight / sample.msample->pdf;
    }

    for(uint i = 0; i < light_samples.size(); i++)
    {
        PathSample sample;
        sample.lsample = &light_samples[i];
        sample.type = PathSample::Light;
        MaterialSample msample;
        sample.msample = &msample;
        msample.wo = (sample.lsample->position - surface.position).normalized();

        if(!material->evaluate_material(surface, msample)) continue;

        const float wo_dot_n = sample.msample->wo.dot(surface.normal);
        Ray sample_ray((wo_dot_n >= 0.f) ? front_position : back_position, sample.msample->wo);
        sample_ray.depth = ray.depth + 1; // Depth should be in the path_sample

        Vec3f in_color = integrate(sample_ray, current_quality, &sample);

        float weight = light_sample_weight;
        if(multiple_importance_sample)
            weight *= (sample.lsample->pdf * sample.lsample->pdf) / ((sample.msample->pdf * sample.msample->pdf) + (sample.lsample->pdf * sample.lsample->pdf));

        color += in_color * sample.msample->fr * weight / sample.lsample->pdf;
    }

    return emission + color;
}

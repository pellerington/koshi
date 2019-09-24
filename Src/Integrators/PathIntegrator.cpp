#include "PathIntegrator.h"

#include <cmath>

#include "../Util/Color.h"


void PathIntegrator::pre_render()
{
}

Vec3f PathIntegrator::integrate(Ray &ray, const float current_quality, float * light_pdf, const LightSample * light_sample) const
{
    // Initialize output
    Vec3f color = VEC3F_ZERO;
    Vec3f emission = VEC3F_ZERO;

    // If we have a light sample set our tmax to light_sample->t
    if(light_sample)
        ray.t = light_sample->t;

    // Intersect with our scene
    Surface surface;
    scene->intersect(ray, surface);

    // Check if we intersected any lights
    if(ray.depth > 0 || scene->settings.display_lights)
        scene->evaluate_lights(ray, emission, light_pdf, light_sample);

    // If we hit nothing end it here (If we have a light_sample and we dont intersect this could incorrectly trigger)
    if(!ray.hit)
    {
        // Evaluate our environment light
        if(ray.t == FLT_MAX)
        {
            Vec3f environment = VEC3F_ZERO;
            scene->evaluate_environment_light(ray, environment);
            if(ray.depth > 0 || scene->settings.display_lights)
                emission += environment;
        }
        return emission;
    }

    // if(!material) Use some default material.
    std::shared_ptr<Material> material = surface.object->material->instance(surface);

    // Add any emissive component of the material to the integrator
    emission += material->get_emission();

    // If we are greater than the max depth, dont perform more integrations
    if(ray.depth > scene->settings.max_depth)
        return emission;

    // Setup sampling variables
    const float sample_multiplier = scene->settings.quality * current_quality;
    std::deque<PathSample> path_samples;

    // Sample the material
    if(scene->settings.sample_material)
        material->sample_material(surface, path_samples, sample_multiplier);
    const uint n_material_samples = path_samples.size();

    // Sample the lights
    if(scene->settings.sample_lights)
        scene->sample_lights(surface, path_samples, sample_multiplier);
    const uint n_light_samples = path_samples.size() - n_material_samples;

    // Return if we have no samples
    if(!(n_material_samples + n_light_samples))
        return emission;

    const bool multiple_importance_sample = (n_light_samples > 0) && (n_material_samples > 0);
    const float material_sample_weight = (n_material_samples > 0) ? 1.f / n_material_samples : 0.f;
    const float light_sample_weight = (n_light_samples > 0) ? 1.f / n_light_samples : 0.f;

    for(;!path_samples.empty(); path_samples.pop_front())
    {
        PathSample &sample = path_samples.front();

        float mis_pdf = 0.f;
        float weight = 1.f;

        if(sample.type == PathSample::Light)
        {
            if(!material->evaluate_material(surface, sample, mis_pdf)) continue;
            weight = light_sample_weight;
        }
        else if(sample.type == PathSample::Material)
            weight = material_sample_weight;

        const float wo_dot_n = sample.wo.dot(surface.normal);
        const Vec3f bias_position = surface.position + ((wo_dot_n >= 0.f) ? (surface.normal * EPSILON_F) : (surface.normal * -EPSILON_F)); // Get this from surface
        Ray sample_ray(bias_position, sample.wo);
        sample_ray.depth = ray.depth + 1;

        if(sample.type == PathSample::Material)
            sample.color = integrate(sample_ray, current_quality * sample.quality, &mis_pdf, nullptr);
        else if(sample.type == PathSample::Light)
            sample.color = integrate(sample_ray, current_quality * sample.quality, nullptr, &sample.light_sample);

        if(multiple_importance_sample)
            weight *= (sample.pdf * sample.pdf) / ((sample.pdf * sample.pdf) + (mis_pdf * mis_pdf));

        color += sample.color * sample.fr * weight / sample.pdf;
    }

    return emission + color;
}

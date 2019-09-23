#include "PathIntegrator.h"

#include <cmath>

#include "../Util/Color.h"

Vec3f PathIntegrator::get_color()
{
    return emission + color;
}

void PathIntegrator::pre_render()
{
}

std::shared_ptr<Integrator> PathIntegrator::create(Ray &ray, const float current_quality, float* light_pdf, const LightSample* light_sample)
{
    std::shared_ptr<PathIntegrator> path_integrator(new PathIntegrator(scene));
    path_integrator->setup(ray, current_quality, light_pdf, light_sample);
    return path_integrator;
}

void PathIntegrator::setup(Ray &ray, const float _current_quality, float* light_pdf, const LightSample* light_sample)
{
    // Initialize variables
    color = VEC3F_ZERO;
    normalization = 0.f;
    emission = VEC3F_ZERO;
    depth = ray.depth;
    current_quality = _current_quality;

    // If we have a light sample set our tmax to light_sample->t
    if(light_sample)
        ray.t = light_sample->t;

    // Intersect with our scene
    scene->intersect(ray, surface);

    // Check if we intersected any lights
    if(depth > 0 || scene->settings.display_lights)
        scene->evaluate_lights(ray, emission, light_pdf, light_sample);

    // If we hit nothing end it here
    if(!surface.object || !ray.hit)
    {
        Vec3f environment = 0.f;
        scene->evaluate_environment_light(ray, environment);
        if(depth > 0 || scene->settings.display_lights)
            emission += environment;
        return;
    }

    // if(!material)
        // Use some default material.
    material = surface.object->material->instance(surface);

    // Add any emissive component of the material to the integrator
    emission += material->get_emission();

    // If we are greater than the max depth, dont perform integration
    if(depth > scene->settings.max_depth)
        return;

    // Reduce number of samples
    const float sample_multiplier = scene->settings.quality * current_quality;

    // Sample the material
    if(scene->settings.sample_material)
        material->sample_material(surface, path_samples, sample_multiplier);
    const uint n_material_samples = path_samples.size();

    // Sample the lights
    if(scene->settings.sample_lights)
        scene->sample_lights(surface, path_samples, sample_multiplier);
    const uint n_light_samples = path_samples.size() - n_material_samples;

    multiple_importance_sample = (n_light_samples > 0) && (n_material_samples > 0);
    material_sample_weight = (n_material_samples > 0) ? 1.f / n_material_samples : 0.f;
    light_sample_weight = (n_light_samples > 0) ? 1.f / n_light_samples : 0.f;

    // Shuffle the samples if it is for a pixel
    if(!depth)
        RNG::Shuffle<PathSample>(path_samples);
}

void PathIntegrator::integrate(size_t num_samples)
{
    for(size_t i = 0; i < num_samples && !path_samples.empty(); path_samples.pop_front(), i++)
    {
        PathSample &sample = path_samples.front();

        const float wo_dot_n = sample.wo.dot(surface.normal);

        float mis_pdf = 0.f;
        float weight = 1.f;

        if(sample.type == PathSample::Light)
        {
            if(!material->evaluate_material(surface, sample, mis_pdf))
                continue;
            weight = light_sample_weight;
        }
        else if(sample.type == PathSample::Material)
            weight = material_sample_weight;

        Ray ray;
        ray.o = surface.position + ((wo_dot_n >= 0.f) ? (surface.normal * EPSILON_F) : (surface.normal * -EPSILON_F));
        ray.dir = sample.wo;
        ray.depth = depth + 1;

        std::shared_ptr<Integrator> integrator = (sample.type == PathSample::Material)
            ? create(ray, current_quality * sample.quality, &mis_pdf, nullptr)
            : create(ray, current_quality * sample.quality, nullptr, &sample.light_sample);
        integrator->integrate(integrator->get_required_samples());

        if(multiple_importance_sample)
            weight *= (sample.pdf * sample.pdf) / ((sample.pdf * sample.pdf) + (mis_pdf * mis_pdf));

        sample.color = integrator->get_color();
        color += sample.color * sample.fr * weight / sample.pdf;

        normalization = normalization + 1.f;
    }
}

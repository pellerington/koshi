#include "PathIntegrator.h"

#include <cmath>

#include "../Util/Color.h"

const Vec3f PathIntegrator::get_color()
{
    return emission + color;
}

void PathIntegrator::pre_render()
{
    scene->accelerate();
}

std::shared_ptr<Integrator> PathIntegrator::create(Ray &ray, float* light_pdf, const LightSample* light_sample)
{
    std::shared_ptr<PathIntegrator> path_integrator(new PathIntegrator(scene));
    path_integrator->setup(ray, light_pdf, light_sample);
    return path_integrator;
}

void PathIntegrator::setup(Ray &ray, float* light_pdf, const LightSample* light_sample)
{
    // Initialize variables
    color = Vec3f::Zero();
    normalization = 0.f;
    emission = Vec3f::Zero();
    depth = ray.depth;

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
        emission += environment;
        return;
    }

    material = surface.object->material; // This should be surface.object->material->get_material(Surface surface); (So that textures can be evaluated once.)

    // Add any emissive component of the material to the integrator
    emission += material->get_emission();

    // If we are greater than the max depth, dont perform integration
    if(depth > scene->settings.max_depth)
        return;

    // Reduce number of samples base on the depth
    float sample_multiplier = scene->settings.quality / powf((depth + 1), 4);

    // Sample the material
    if(scene->settings.sample_material)
        material->sample_material(surface, srf_samples, sample_multiplier);
    const uint n_material_samples = srf_samples.size();

    // Sample the lights
    if(scene->settings.sample_lights)
        scene->sample_lights(surface, srf_samples, sample_multiplier);
    const uint n_light_samples = srf_samples.size() - n_material_samples;

    multiple_importance_sample = (n_light_samples > 0) && (n_material_samples > 0);
    material_sample_weight = (n_material_samples > 0) ? 1.f / n_material_samples : 0.f;
    light_sample_weight = (n_light_samples > 0) ? 1.f / n_light_samples : 0.f;

    // Shuffle the samples if it is for a pixel
    if(!depth)
        RNG::Shuffle<SrfSample>(srf_samples);
}

void PathIntegrator::integrate(size_t num_samples)
{
    for(size_t i = 0; i < num_samples && !srf_samples.empty(); srf_samples.pop_front(), i++)
    {
        SrfSample &sample = srf_samples.front();

        float mis_pdf = 0.f;
        float weight = 1.f;
        if(sample.type == SrfSample::Light)
        {
            if(!material->evaluate_material(surface, sample, mis_pdf))
                continue;
            weight = light_sample_weight;
        }
        else if(sample.type == SrfSample::Material)
            weight = material_sample_weight;

        Ray ray;
        ray.o = surface.position;
        ray.dir = sample.wo;
        ray.depth = depth + 1;
        std::shared_ptr<Integrator> integrator = (sample.type == SrfSample::Material) ? create(ray, &mis_pdf, nullptr) : create(ray, nullptr, &sample.light_sample);
        integrator->integrate(integrator->get_required_samples());

        if(multiple_importance_sample)
            weight *= (sample.pdf * sample.pdf) / ((sample.pdf * sample.pdf) + (mis_pdf * mis_pdf));

        sample.color = integrator->get_color();
        color += sample.color * sample.fr * sample.wo.dot(surface.normal) * weight / sample.pdf;

        normalization = normalization + 1.f;
    }
}

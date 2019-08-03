#include "Scene.h"

void Scene::accelerate()
{
    std::vector<std::shared_ptr<Object>> sub_objects;
    for(size_t i = 0; i < objects.size(); i++)
    {
        std::vector<std::shared_ptr<Object>> i_sub_objects = objects[i]->get_sub_objects();
        sub_objects.insert(sub_objects.end(), i_sub_objects.begin(), i_sub_objects.end());
    }
    accelerator = std::unique_ptr<Accelerator>(new Accelerator(sub_objects));
}

bool Scene::intersect(Ray &ray, Surface &surface)
{
    if(accelerator)
    {
        return accelerator->intersect(ray, surface);
    }
    else
    {
        for(size_t i = 0; i < objects.size(); i++)
        {
            objects[i]->intersect(ray, surface);
        }
        return ray.hit;
    }
}

inline void Scene::evaluate_light(const uint i, const Ray &ray, Vec3f &light, float* pdf)
{
    Vec3f light_i = 0.f;
    float pdf_i = 0.f;
    lights[i]->evaluate_light(ray, light_i, &pdf_i);
    light += light_i;
    if(pdf) *pdf += pdf_i;
}

bool Scene::evaluate_lights(const Ray &ray, Vec3f &light, float* pdf, const LightSample* light_sample)
{
    light = 0.f;
    if(pdf) *pdf = 0.f;
    if(light_sample)
    {
        for(uint i = 0; i < light_sample->id; i++)
            evaluate_light(i, ray, light, pdf);
        if(ray.t >= light_sample->t)
            light += light_sample->intensity;
        for(uint i = light_sample->id + 1; i < lights.size(); i++)
            evaluate_light(i, ray, light, pdf);
    }
    else
    {
        for(uint i = 0; i < lights.size(); i++)
            evaluate_light(i, ray, light, pdf);
    }
    return true;
}

bool Scene::evaluate_environment_light(const Ray &ray, Vec3f &light, float* pdf)
{
    if(environment_light)
        return environment_light->evaluate_light(ray, light, pdf);
    return false;
}

bool Scene::sample_lights(const Surface &surface, std::deque<SrfSample> &srf_samples, const float sample_multiplier)
{
    for(size_t i = 0; i < lights.size(); i++)
    {
        const uint num_samples = std::max(1.f, lights[i]->estimated_samples(surface) * sample_multiplier); // Do the max inside the light?
        std::deque<LightSample> light_samples;
        lights[i]->sample_light(num_samples, surface, light_samples);
        for(uint j = 0; j < light_samples.size(); j++)
        {
            srf_samples.emplace_back();
            SrfSample &srf_sample = srf_samples.back();
            srf_sample.type = SrfSample::Light;

            Vec3f dir = light_samples[j].position - surface.position;
            srf_sample.wo = dir.normalized();
            srf_sample.pdf = light_samples[j].pdf;

            srf_sample.light_sample = light_samples[j];
            srf_sample.light_sample.id = i;
            srf_sample.light_sample.t = dir.length();
        }
    }
    return true;
}

bool Scene::add_object(std::shared_ptr<Object> object)
{
    objects.push_back(object);
    return true;
}

bool Scene::add_material(std::shared_ptr<Material> material)
{
    materials.push_back(material);
    return true;
}

bool Scene::add_light(std::shared_ptr<Light> light)
{
    if(light->type == Light::Environment)
        environment_light = light;
    else
        lights.push_back(light);
    return true;
}

bool Scene::add_texture(std::shared_ptr<Texture> texture)
{
    textures.push_back(texture);
    return true;
}

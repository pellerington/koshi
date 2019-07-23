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

bool Scene::evaluate_lights(const Ray &ray, Vec3f &light, float* pdf)
{
    light = 0.f;
    if(pdf)
        *pdf = 0.f;
    for(size_t i = 0; i < lights.size(); i++)
    {
        Vec3f l = 0.f;
        float p = 0.f;
        lights[i]->evaluate_light(ray, l, &p);
        light += l;
        if(pdf)
            *pdf += p;
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
        const uint num_samples = std::max(1.f, lights[i]->estimated_samples(surface) * sample_multiplier); // Do the max inside the light
        lights[i]->sample_light(num_samples, surface, srf_samples);
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

#include <Scene/Scene.h>

void Scene::pre_render()
{
}

void Scene::sample_lights(const Intersect& intersect, std::vector<LightSample> &light_samples, const float sample_multiplier, Resources &resources)
{
    for(size_t i = 0; i < lights.size(); i++)
    {
        // Make a better num_samples estimator
        const uint num_samples = std::max(1.f, SAMPLES_PER_SA * sample_multiplier);
        LightSampler * sampler = lights[i]->get_attribute<LightSampler>("light_sampler");
        if(sampler)
            sampler->sample_light(num_samples, intersect, light_samples, resources);
    }
}

bool Scene::add_object(std::shared_ptr<Geometry> object)
{
    objects.push_back(object);
    return true;
}

bool Scene::add_material(std::shared_ptr<Material> material)
{
    materials.push_back(material);
    return true;
}

bool Scene::add_light(std::shared_ptr<Geometry> light)
{
    lights.push_back(light);
    add_object(light);
    return true;
}

bool Scene::add_texture(std::shared_ptr<Texture> texture)
{
    textures.push_back(texture);
    return true;
}

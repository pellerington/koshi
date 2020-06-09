#include <base/Scene.h>

void Scene::pre_render(Resources& resources)
{
    for(auto it = objects.begin(); it != objects.end(); it++)
        it->second->pre_render(resources);
}

bool Scene::add_object(const std::string& name, Object * object)
{
    objects.insert(std::make_pair(name, object));
    return true;
}

Object * Scene::get_object(const std::string& name)
{
    auto object = objects.find(name);
    if(object != objects.end())
        return object->second;
    return nullptr;
}

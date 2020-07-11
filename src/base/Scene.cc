#include <base/Scene.h>

void Scene::pre_render(Resources& resources)
{
    for(auto it = objects.begin(); it != objects.end(); it++)
        it->second->pre_render(resources);
}
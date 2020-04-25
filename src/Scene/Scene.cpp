#include <Scene/Scene.h>

void Scene::pre_render()
{
    for(uint i = 0; i < objects.size(); i++)
        objects[i]->pre_render(this);
}

bool Scene::add_object(Object * object)
{
    objects.push_back(object);
    return true;
}
#include <Scene/Scene.h>

#include <integrators/Integrator.h>
void Scene::pre_render()
{
    // TODO: Remove this when we move to the other pre_render type.
    objects[0]->get_attribute<Integrator>("integrator")->pre_render(this);
}

bool Scene::add_object(Geometry * object)
{
    objects.push_back(object);
    return true;
}

bool Scene::add_material(Material * material)
{
    materials.push_back(material);
    return true;
}

bool Scene::add_texture(Texture * texture)
{
    textures.push_back(texture);
    return true;
}

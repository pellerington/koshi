#include <intersection/Intersector.h>

#include <geometry/Geometry.h>
#include <Scene/Scene.h>

Intersector::Intersector(Scene * scene)
: scene(scene)
{
    std::vector<std::shared_ptr<Geometry>>& objects = scene->get_objects();
    for(size_t i = 0; i < objects.size(); i++)
    {
        IntersectionCallbacks * callbacks = objects[i]->get_attribute<IntersectionCallbacks>("intersection_callbacks");
        if(callbacks)
        {
            if(callbacks->null_intersection_cb)
                null_callbacks.push_back(std::make_pair(callbacks->null_intersection_cb, objects[i].get()));
        }
    }
}
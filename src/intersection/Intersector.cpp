#include <intersection/Intersector.h>

#include <geometry/Geometry.h>
#include <Scene/Scene.h>

Intersector::Intersector(Scene * scene)
: scene(scene)
{
    std::vector<Object*>& objects = scene->get_objects();
    for(size_t i = 0; i < objects.size(); i++)
    {
        Geometry * geometry = dynamic_cast<Geometry*>(objects[i]);
        if(geometry)
        {
            IntersectionCallbacks * callbacks = geometry->get_attribute<IntersectionCallbacks>("intersection_callbacks");
            if(callbacks)
            {
                if(callbacks->null_intersection_cb)
                    null_callbacks.push_back(std::make_pair(callbacks->null_intersection_cb, geometry));
            }
        }
    }
}
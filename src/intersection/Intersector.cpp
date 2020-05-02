#include <intersection/Intersector.h>

#include <geometry/Geometry.h>
#include <base/Scene.h>

Intersector::Intersector(Scene * scene)
: scene(scene)
{
    for(auto object = scene->begin(); object != scene->end(); ++object)
    {
        Geometry * geometry = dynamic_cast<Geometry*>(object->second);
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
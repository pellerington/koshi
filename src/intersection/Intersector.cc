#include <intersection/Intersector.h>
#include <geometry/Geometry.h>

Intersector::Intersector(ObjectGroup * objects)
{
    for(uint i = 0; i < objects->size(); i++)
    {
        Geometry * geometry = objects->get<Geometry>(i);
        if(geometry)
        {
            IntersectionCallbacks * callback = geometry->get_attribute<IntersectionCallbacks>("intersection_callbacks");
            if(callback) callbacks.push_back(callback);
        }
    }
}
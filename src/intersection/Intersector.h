#pragma once

#include <intersection/Intersect.h>
#include <intersection/Ray.h>
class Scene;

class Intersector
{
public:
    Intersector(Scene * scene);

    virtual void pre_render() = 0;

    virtual Intersect intersect(Ray& ray) = 0;
    
    void null_intersection_callbacks(Intersect& intersect)
    {
        for(size_t i = 0; i < null_callbacks.size(); i++)
            null_callbacks[i].first(intersect, null_callbacks[i].second);
    }

protected:
    Scene * scene;

    std::vector<std::pair<NullIntersectionCallback*, Geometry*>> null_callbacks;
};